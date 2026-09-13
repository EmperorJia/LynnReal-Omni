"""Hopper TMA GEMM with INT32 accumulation and the reference BF16 epilogue.

Scheduling follows Triton's persistent-matmul tutorial (v3.3.1); this module
is selected only for measured long-sequence shapes; other GPUs use int8_gemm.
"""
import torch
import triton
import triton.language as tl
from triton.tools.experimental_descriptor import create_2d_tma_descriptor

@triton.jit
def _persistent(A, B, SA, SB, Bias, C, M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                BIAS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SMS:tl.constexpr):
    nm,nn=tl.cdiv(M,BM),tl.cdiv(N,BN)
    for tile in tl.range(tl.program_id(0),nm*nn,SMS):
        group=tile//(8*nn)
        size=tl.minimum(nm-group*8,8)
        mi=group*8+(tile%(8*nn))%size
        ni=(tile%(8*nn))//size
        acc=tl.full((BM,BN),0,tl.int32)
        for ki in range(tl.cdiv(K,BK)):
            a=tl._experimental_descriptor_load(A,[mi*BM,ki*BK],[BM,BK],tl.int8)
            b=tl._experimental_descriptor_load(B,[ni*BN,ki*BK],[BN,BK],tl.int8)
            acc=tl.dot(a,b.T,acc)
        rows=mi.to(tl.int64)*BM+tl.arange(0,BM)
        cols=ni*BN+tl.arange(0,BN)
        value=acc.to(tl.float32)*tl.load(SA+rows,rows<M,other=0)[:,None]
        value=value*tl.load(SB+cols,cols<N,other=0)[None,:]
        if BIAS:value=value+tl.load(Bias+cols,cols<N,other=0).to(tl.float32)[None,:]
        tl.store(C+rows[:,None]*N+cols[None,:],value,(rows[:,None]<M)&(cols[None,:]<N))

def matmul(a,b,sa,sb,bias=None,tile=(128,256,128,8,3)):
    m,k=a.shape;n=b.shape[1];bm,bn,bk,warps,stages=tile
    if a.dtype!=torch.int8 or b.dtype!=torch.int8 or not a.is_contiguous() or b.stride()!=(1,k):
        raise ValueError('TMA requires row-major INT8 activations and transposed row-major weights')
    da=create_2d_tma_descriptor(a.data_ptr(),m,k,bm,bk,1)
    db=create_2d_tma_descriptor(b.data_ptr(),n,k,bn,bk,1)
    out=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
    sms=torch.cuda.get_device_properties(a.device).multi_processor_count
    _persistent[(min(sms,triton.cdiv(m,bm)*triton.cdiv(n,bn)),)](da,db,sa,sb,bias,out,m,n,k,bias is not None,bm,bn,bk,sms,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return out
