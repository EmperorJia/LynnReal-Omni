"""Compile and numerically check optional kernels on the installation GPU."""
import argparse
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_ordered_qk(report, rng):
    import torch
    from model.kernels import _qk_norm_rope_kernel
    # Include batched, strided Q/K and a non-tile-aligned sequence.
    for batch, tokens in ((1, 128), (2, 137)):
        qkv = torch.randn(batch, tokens, 12, 128, device='cuda', dtype=torch.bfloat16, generator=rng)
        q, k = qkv[:, :, :4], qkv[:, :, 4:8]
        weights = [torch.randn(128, device='cuda', dtype=q.dtype, generator=rng) for _ in range(2)]
        phase = torch.randn(tokens, 16, device='cuda', generator=rng)
        outputs = []
        for ordered in (False, True):
            qo = torch.empty(q.shape, device=q.device, dtype=q.dtype)
            ko = torch.empty_like(qo)
            _qk_norm_rope_kernel[(batch*tokens*4,)](
                q, k, *weights, phase.cos(), phase.sin(), qo, ko, tokens, 4,
                q.stride(1), q.stride(2), k.stride(1), k.stride(2), 128, 16, 1e-5, 128,
                ordered_reduction=ordered, num_warps=1 if ordered else 4, enable_fp_fusion=False)
            outputs.append((qo, ko))
        exact = all(torch.equal(a, b) for a, b in zip(*outputs))
        report['checks'].append({'kernel': 'ordered_qk', 'batch': batch, 'tokens': tokens, 'exact': exact})
        if not exact:
            raise AssertionError('Ordered Q/K does not match the reference reduction')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, default=ROOT/'output/setup/kernels.json')
    args = p.parse_args()
    import torch
    report = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'checks': []}
    if torch.cuda.is_available():
        from model.acceleration import fastest_available
        report.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
        with torch.inference_mode(), torch.random.fork_rng():
            try:
                report['acceleration'] = fastest_available()
                from model.int8_gemm import int8_matmul, _torch_matmul
                rng = torch.Generator(device='cuda').manual_seed(219)
                shapes = [(32,21504,5376), (32,5376,7168), (32,28672,5376), (32,5376,14336)]
                if torch.cuda.get_device_capability()[0] == 9:
                    shapes += [(16385,21504,5376), (16385,5376,7168),
                               (49153,28672,5376), (16385,5376,14336)]
                    check_ordered_qk(report, rng)
                for m,n,k in shapes:
                    a=torch.randint(-127,128,(m,k),device='cuda',dtype=torch.int8,generator=rng)
                    b=torch.randint(-127,128,(n,k),device='cuda',dtype=torch.int8,generator=rng).t()
                    rs=torch.rand(m,1,device='cuda',generator=rng)/127
                    cs=torch.rand(1,n,device='cuda',generator=rng)/127
                    actual=int8_matmul(a,b,rs,cs)
                    exact=torch.equal(actual,_torch_matmul(a,b,rs,cs,None))
                    report['checks'].append({'shape':[m,n,k], 'exact':exact})
                    if not exact:
                        raise AssertionError('INT8 kernel did not match integer reference')
                from types import SimpleNamespace
                from model.kernels import residual_rms_quant, _gated_residual, _rms_adaln_quant_int8
                x=torch.randn(1,32,5376,device='cuda',dtype=torch.bfloat16,generator=rng)
                update=torch.randn(x.shape,device='cuda',dtype=x.dtype,generator=rng)
                table=torch.randn(2,6*5376,device='cuda',dtype=x.dtype,generator=rng)
                indices=torch.arange(32,device='cuda')%2
                norm=SimpleNamespace(weight=torch.ones(5376,device='cuda',dtype=x.dtype),eps=1e-5)
                residual=_gated_residual(x,update,table,indices,2)
                expected=(residual,*_rms_adaln_quant_int8(residual,norm,table,indices,3,4))
                actual=residual_rms_quant(x,update,norm,table,indices)
                exact=all(torch.equal(a,b) for a,b in zip(actual,expected))
                report['checks'].append({'kernel':'residual_rms_quant','exact':exact})
                if not exact:raise AssertionError('residual fusion did not match separate kernels')
                torch.cuda.synchronize()
                report['status']='passed' 
            except Exception as error:
                report.update(status='failed' if isinstance(error, AssertionError) else 'unavailable', reason=str(error))
                warnings.warn('Optional kernel precompilation unavailable: '+str(error))
    else:
        report.update(status='deferred', reason='No visible CUDA GPU; kernels compile on first use.')
        warnings.warn(report['reason'])
    report['scope']='This GPU and representative projections; new shapes may require compilation and tuning.'
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if report['status'] == 'failed':
        print('Kernel numerical validation failed; see '+str(args.report), file=sys.stderr)
        raise SystemExit(2)


if __name__=='__main__':main()
