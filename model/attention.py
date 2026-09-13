"""Device-tested exact-attention backends; probes never advance sampling RNG."""
import os
import warnings
import torch

BACKENDS = ('_flash_3', 'flash', '_native_cudnn', '_native_flash', 'native')


@torch.inference_mode()
def probe_backend(backend):
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if backend not in BACKENDS:
        raise ValueError('unknown attention backend: ' + backend)
    errors = []
    generator = torch.Generator(device='cuda').manual_seed(91827)
    for dtype in (torch.bfloat16, torch.float16):
        for dim in (64, 96, 128):
            q = torch.randn(1,129,4,dim,device='cuda',dtype=dtype,generator=generator)
            k = torch.randn(1,77,4,dim,device='cuda',dtype=dtype,generator=generator)
            v = torch.randn(1,77,4,dim,device='cuda',dtype=dtype,generator=generator)
            with sdpa_kernel(SDPBackend.MATH):
                reference = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1,2).float(),k.transpose(1,2).float(),v.transpose(1,2).float()).transpose(1,2)
            actual = dispatch_attention_fn(q,k,v,dropout_p=0.0,is_causal=False,backend=backend)
            torch.cuda.synchronize()
            diff=actual.float()-reference
            relative=float(diff.square().mean().sqrt()/reference.square().mean().sqrt().clamp_min(1e-8))
            if not torch.isfinite(actual).all() or relative > 0.02:
                raise RuntimeError(f'{backend}: attention numerical check failed ({relative})')
            errors.append({'dtype':str(dtype), 'head_dim':dim,'relative_l2':relative,'max_abs':float(diff.abs().max())})
    return {'backend':backend,'checks':errors}


def select_attention(requested=None):
    capability=torch.cuda.get_device_capability()
    requested=requested or os.environ.get('LYNNREAL_ATTENTION','auto')
    candidates=list(BACKENDS) if requested=='auto' else [requested]
    failures=[]
    for backend in candidates:
        if backend=='_flash_3' and capability[0]!=9:continue
        if backend=='flash' and capability[0]<8:continue
        try:
            check = probe_backend(backend)
            if backend != '_flash_3':
                warnings.warn(f'FA3 is not active; using {backend}. Attention backends can change '
                              'speed and sampled results; verify videos before comparing them.', RuntimeWarning)
            return backend,failures,check
        except Exception as error:
            if isinstance(error,torch.cuda.OutOfMemoryError):raise
            failures.append(f'{backend}: {error}')
    raise RuntimeError('No supported attention backend: '+'; '.join(failures))
