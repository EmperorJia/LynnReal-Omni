"""Probe optional CUDA kernels before loading models; never retry failed inference."""
import torch


def fastest_available(attention_backend=None):
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling requires a CUDA GPU")
    result = {"attention_backend": "native", "fused": False, "fallbacks": []}
    capability = torch.cuda.get_device_capability()
    with torch.inference_mode(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        from .attention import select_attention
        backend, failures, check = select_attention(attention_backend)
        result.update(attention_backend=backend, attention_numerical_check=check)
        result["fallbacks"].extend(failures)
        try:
            if capability[0] < 8:
                raise RuntimeError("BF16 fusion requires compute capability >= 8.0")
            from types import SimpleNamespace
            from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
            from .fusion import enable_fusion
            block = MiniMaxH3TransformerBlock(192, 2, 96, 384, 192, 1e-5, 1e-5)
            block = block.to(device="cuda", dtype=torch.bfloat16).eval()
            block.attn.processor._attention_backend = result["attention_backend"]
            enable_fusion(SimpleNamespace(transformer_blocks=[block]))
            x = torch.ones(1, 16, 192, device="cuda", dtype=torch.bfloat16)
            temb = torch.zeros(1, 192, device="cuda", dtype=torch.bfloat16)
            indices = torch.zeros(16, device="cuda", dtype=torch.long)
            for rotary_dim in (48, 96):
                cos = torch.ones(16, rotary_dim, device="cuda")
                block(x, temb, indices, (cos, torch.zeros_like(cos)))
            from .kernels import residual_rms_quant, _gated_residual, _rms_adaln_quant_int8
            update=torch.randn_like(x)
            table=torch.randn(1,6*x.shape[-1],device=x.device,dtype=x.dtype)
            residual=_gated_residual(x,update,table,indices,2)
            expected=(residual,*_rms_adaln_quant_int8(residual,block.norm2,table,indices,3,4))
            actual=residual_rms_quant(x,update,block.norm2,table,indices)
            if not all(torch.equal(a,b) for a,b in zip(actual,expected)):
                raise RuntimeError('Residual fusion failed its exact numerical probe')
            torch.cuda.synchronize()
            result["fused"] = True
            result["residual_quant_exact"] = True
        except Exception as error:
            if isinstance(error, torch.cuda.OutOfMemoryError):
                raise
            result["fallbacks"].append(f"Triton fusion unavailable: {error}")
    return result


def optional_kernel_failure(error):
    """Compilation/device support failures are recoverable; corrupt CUDA state is not."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return False
    module = type(error).__module__
    return (module.startswith(('triton', 'torch._dynamo', 'torch._inductor'))
            or any(s in str(error).lower() for s in
                   ('not supported', 'not implemented', 'no kernel image', 'invalid device function')))


def compile_decoder(decoder):
    """Fall back before decoding completes if optional compilation is unsupported."""
    import warnings
    native = decoder.forward
    compiled = torch.compile(native, fullgraph=True, dynamic=False)
    report = dict(backend='inductor', mode='default', fullgraph=True, dynamic=False)

    def forward(*args, **kwargs):
        try:
            return compiled(*args, **kwargs)
        except Exception as error:
            if not optional_kernel_failure(error):
                raise
            warnings.warn(f'Decoder compilation unavailable; using eager decoder: {error}', RuntimeWarning)
            decoder.forward = native
            report.update(backend='eager', fallback=str(error))
            return native(*args, **kwargs)

    decoder.forward = forward
    return report
