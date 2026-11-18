"""GEMM v0 的数值正确性测试。

注意：包名从 kernels 改成 baseline —— `kernels` 是 HuggingFace 的 PyPI 包名，
在装过 transformers 生态的机器上会被 site-packages 抢走（实测在 H100 机器上炸过）。

- 本地 macOS：import triton 失败 -> SkipTest（这是预期行为，不是失败）
- Linux 容器：   TRITON_INTERPRET=1 -> 在 CPU 上逐 program 解释执行
- GPU 机器：     真实编译执行

三种环境下是同一份测试代码，这是「本地开发 / 远端验证」双轨的关键设计。
"""

from __future__ import annotations

import unittest

import torch


def _load_triton():
    try:
        import triton  # noqa: F401
    except Exception as e:  # ImportError on macOS，或其它环境问题
        raise unittest.SkipTest(f"triton 不可用（macOS 无轮子）：{e}")
    from baseline.gemm.v0_naive import matmul

    return matmul


def _device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # 解释器模式


def test_naive_matmul_non_multiple_shapes():
    matmul = _load_triton()
    dev = _device()
    M, N, K = 37, 53, 41  # 故意都不是 block 的整数倍，考 mask
    a = torch.randn(M, K, device=dev)
    b = torch.randn(K, N, device=dev)
    ref = a @ b
    out = matmul(a, b, block_m=16, block_n=16, block_k=16)
    assert out.shape == (M, N)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (
        (out - ref).abs().max().item()
    )


def test_naive_matmul_exact_blocks():
    matmul = _load_triton()
    dev = _device()
    M, N, K = 64, 32, 32
    a = torch.randn(M, K, device=dev)
    b = torch.randn(K, N, device=dev)
    out = matmul(a, b, block_m=32, block_n=32, block_k=32)
    assert torch.allclose(out, a @ b, atol=1e-5, rtol=1e-5)
