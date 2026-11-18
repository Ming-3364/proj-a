"""GEMM v0（naive）：每个 program 一个 BLOCK_M×BLOCK_N 输出块，直接读全局内存。

它的作用不是「快」，而是**作为 tiling ladder 的基线**：
  - 没有任何显式的 shared-memory 复用（复用只靠 L1/L2 缓存碰运气）；
  - 全局访存放大倍数 ≈ (BLOCK_N / BLOCK_K) 与 (BLOCK_M / BLOCK_K)；
  - 之后 v1 引入 smem tiling、v2 引入 grouped ordering，都用它作对照。

注意：本地 macOS 无法 import triton（无轮子）；这个模块只在 Linux 容器
（TRITON_INTERPRET=1，CPU 解释器）或 GPU 机器上被导入。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_v0(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k * BLOCK_K + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_off[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_off[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k_off[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_off[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    c = acc.to(tl.float32)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def matmul(a: torch.Tensor, b: torch.Tensor, block_m=64, block_n=64, block_k=32) -> torch.Tensor:
    """a: [M, K]，b: [K, N]（b 已是 K-major，方便直接用 stride 表达转置与否）。"""
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0]
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    matmul_v0[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return c
