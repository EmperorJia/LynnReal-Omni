"""Experimental INT8 tensor-core matmul with a fused scaled BF16 epilogue."""
import hashlib
import json
import os
from pathlib import Path
import warnings

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, AS, BS, Bias, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          B0: tl.constexpr, B1: tl.constexpr, HAS_BIAS: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr = 1):
    if GROUP_M == 0:
        rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
        ni = tl.program_id(1)
    else:
        pid = tl.program_id(0)
        nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
        group = pid // (GROUP_M * nn)
        first = group * GROUP_M
        group_size = tl.minimum(nm - first, GROUP_M)
        mi = first + (pid % (GROUP_M * nn)) % group_size
        ni = (pid % (GROUP_M * nn)) // group_size
        rows = mi * BM + tl.arange(0, BM)
        if M * K >= 2147483648 or M * N >= 2147483648:
            rows = rows.to(tl.int64)
    cols = ni * BN + tl.arange(0, BN)
    depth = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.int32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + depth
        a = tl.load(A + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), other=0)
        b = tl.load(B + kk[:, None] * B0 + cols[None, :] * B1,
                    (kk[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    output = acc.to(tl.float32) * tl.load(AS + rows, rows < M, other=0)[:, None]
    output = output * tl.load(BS + cols, cols < N, other=0)[None, :]
    if HAS_BIAS:
        output = output + tl.load(Bias + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(C + rows[:, None] * N + cols[None, :], output,
             (rows[:, None] < M) & (cols[None, :] < N))


_TILES = [(bm, bn, 128, warps, stages, group)
          for bm, bn, warps, stages in ((64, 128, 4, 3), (128, 128, 8, 3),
                                        (128, 256, 8, 3), (128, 256, 8, 4))
          for group in (0, 1, 8)]
_selected, _disabled, _devices = {}, set(), {}
_tma_disabled = set()
_LONG_KERNELS = os.environ.get("LYNNREAL_LONG_KERNELS", "1") != "0"
_CODE = hashlib.sha256(Path(__file__).read_bytes() +
                       Path(__file__).with_name("int8_tma.py").read_bytes()).hexdigest()


def _cache_file(key):
    root = Path(os.environ.get('TRITON_CACHE_DIR', Path.home()/'.triton/cache'))
    return root/'lynnreal'/('int8-'+hashlib.sha256(repr(key).encode()).hexdigest()+'.json')


def _supported_tiles(configs, named_args, **kwargs):
    hopper = torch.cuda.get_device_capability(named_args['A'].device)[0] == 9
    return [c for c in configs if (c.kwargs['GROUP_M'] > 0) == hopper]


def _grid(m, n, bm, bn, group):
    return ((triton.cdiv(m, bm), triton.cdiv(n, bn)) if group == 0
            else (triton.cdiv(m, bm)*triton.cdiv(n, bn),))


# Autotuning only overwrites C; input tensors and integer arithmetic stay unchanged.
_autotuned = triton.autotune(configs=[
    triton.Config({'BM': bm, 'BN': bn, 'BK': bk, 'GROUP_M': group},
                  num_warps=warps, num_stages=stages)
    for bm, bn, bk, warps, stages, group in _TILES
], key=['M', 'N', 'K', 'B0', 'B1', 'HAS_BIAS'],
    prune_configs_by={'early_config_prune': _supported_tiles})(_gemm)


def _torch_matmul(a, b, row_scale, column_scale, bias):
    rows = a.shape[0]
    pad = max(32, triton.cdiv(rows, 8)*8)-rows
    flat = torch.nn.functional.pad(a, (0, 0, 0, pad)) if pad else a
    acc = torch._int_mm(flat, b)[:rows]
    output = acc.float()*row_scale.reshape(-1, 1).float()*column_scale.float()
    if bias is not None:
        output = output+bias.float()
    return output.to(torch.bfloat16)



def int8_matmul(a, b, row_scale, column_scale, bias=None, *, tile=None):
    if a.dtype != torch.int8 or b.dtype != torch.int8 or not a.is_contiguous() or a.shape[1] != b.shape[0]:
        raise ValueError("INT8 GEMM requires contiguous row-major activations and compatible INT8 weights")
    m, k = a.shape
    n = b.shape[1]
    prefix = _devices.get(a.device)
    if prefix is None:
        device = torch.cuda.get_device_properties(a.device)
        prefix = (_CODE, triton.__version__, torch.version.cuda, device.name,
                  device.major, device.minor, device.multi_processor_count)
        _devices[a.device] = prefix
    key = prefix + (m, n, k, b.stride(), bias is not None)
    # TMA helps long Hopper projections; short and unsupported shapes retain
    # the measured autotuned path. The INT32 accumulation/epilogue is unchanged.
    if (_LONG_KERNELS and tile is None and prefix[4] == 9 and m >= 16384
            and (n != 28672 or m >= 49152) and k % 128 == 0
            and b.stride() == (1, k) and key not in _tma_disabled):
        try:
            from .int8_tma import matmul
            return matmul(a, b, row_scale, column_scale, bias)
        except Exception as error:
            from .acceleration import optional_kernel_failure
            if not isinstance(error, (ImportError, AttributeError, ValueError)) and not optional_kernel_failure(error):
                raise
            warnings.warn(f'Optional Hopper TMA unavailable; using reference INT8 GEMM: {error}')
            _tma_disabled.add(key)
    if key in _disabled:
        return _torch_matmul(a, b, row_scale, column_scale, bias)
    explicit = tile is not None
    if tile is None:
        tile = _selected.get(key)
        if tile is None:
            try:
                saved = tuple(json.loads(_cache_file(key).read_text()))
                if saved in _TILES:
                    tile = _selected[key] = saved
            except (OSError, ValueError, TypeError):
                pass
    output = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    try:
        if tile is None:
            _autotuned[lambda meta: _grid(m,n,meta['BM'],meta['BN'],meta['GROUP_M'])](
                a, b, row_scale, column_scale, bias, output, m, n, k,
                b.stride(0), b.stride(1), bias is not None, enable_fp_fusion=False)
            best = _autotuned.best_config
            tile = tuple(best.kwargs[x] for x in ('BM', 'BN', 'BK')) + (
                best.num_warps, best.num_stages, best.kwargs['GROUP_M'])
            _selected[key] = tile
            try:
                path = _cache_file(key)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f'.{os.getpid()}.tmp')
                tmp.write_text(json.dumps(tile))
                tmp.replace(path)
            except OSError:
                pass  # A read-only cache must not prevent sampling.
        else:
            bm, bn, bk, warps, stages, *group = tile
            _gemm[_grid(m,n,bm,bn,group[0] if group else 1)](a, b, row_scale, column_scale, bias, output,
                m, n, k, b.stride(0), b.stride(1), bias is not None, bm, bn, bk, group[0] if group else 1,
                num_warps=warps, num_stages=stages, enable_fp_fusion=False)
        return output
    except Exception as error:
        if explicit or isinstance(error, torch.cuda.OutOfMemoryError):
            raise
        warnings.warn(f'INT8 Triton kernel unavailable; using integer PyTorch GEMM: {error}', RuntimeWarning)
        _disabled.add(key)
        return _torch_matmul(a, b, row_scale, column_scale, bias)
