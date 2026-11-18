"""Roofline：先判断「这道题是带宽题还是算力题」，再谈优化。

关键结论（Llama-3-8B 类，fp16，32 层，8 KV head，head_dim=128）：

    每 token 每层 KV = 2·8·128·2 B = 4,096 B
    每 token 全部层  = 4,096·32     = 131,072 B = 128 KiB
    4K 上下文单序列  = 128 KiB·4096 = 512 MiB        (8K → 1 GiB)
    权重            = 8.03e9·2 B   ≈ 16.1 GB

    → bs=1, 4K：一步 decode 读 ~16.1 GB 权重（H100 PCIe 2.0 TB/s ≈ 8.0 ms 下界）
    → 交叉点与带宽无关：bs·KV_per_seq ≈ 权重 → 4K 时 bs≈31，8K 时 bs≈16

所以 bs<16 的 decode 本质是「权重带宽题」，kernel 优化的可见度极低。
实验必须落在 bs≥32 / ctx≥8K 的 KV-bound 区。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from analysis.hardware import H100_PCIE, DeviceSpec

__all__ = [
    "kv_bytes_per_token",
    "weights_bytes",
    "decode_lower_bound_ms",
    "crossover_batch",
    "split_kv_blocks",
    "efficiency",
    "LLAMA3_8B",
]


def kv_bytes_per_token(
    layers: int, kv_heads: int, head_dim: int, dtype_bytes: int = 2
) -> int:
    """每生成一个 token，KV cache 需要被读取的字节数（K 和 V 各一份）。"""
    return layers * 2 * kv_heads * head_dim * dtype_bytes


def weights_bytes(params: float, dtype_bytes: int = 2) -> int:
    return int(params * dtype_bytes)


def decode_lower_bound_ms(
    batch: int,
    seq_len: int,
    weights_b: int,
    kv_per_token_b: int,
    dev: DeviceSpec = H100_PCIE,
) -> float:
    """单步 decode 的 HBM 时间下界（ms）。

    访存 = 权重（每步读一遍）+ batch·seq 的 KV。
    忽略 block_table 等小项。
    """
    total = weights_b + batch * seq_len * kv_per_token_b
    return total / dev.bandwidth_bps * 1e3


def crossover_batch(weights_b: int, seq_len: int, kv_per_token_b: int) -> float:
    """KV 读取量追平权重读取量时的 batch size（与带宽无关）。"""
    return weights_b / (seq_len * kv_per_token_b)


def split_kv_blocks(
    batch: int,
    seq_len: int,
    block_n: int,
    dev: DeviceSpec = H100_PCIE,
    ctas_per_sm: int = 1,
    max_splits: int = 32,
) -> dict:
    """split-KV 的并行度启发式（不是最优解，是可解释的起点）。

    动机：decode 时 batch 小、BLOCK_M 小，朴素分块后总 block 数可能远小于
    可用 SM 槽位（114·ctas_per_sm），GPU 大面积闲置。
    代价：每个 split 要写 partial (max, sum, acc)，再做一次 merge。

    返回 dict：total_blocks / target_blocks / splits / waste_ratio_before/after
    """
    blocks = batch * math.ceil(seq_len / block_n)
    target = dev.sm_count * ctas_per_sm
    splits = 1
    if blocks > 0 and blocks < target:
        splits = min(max_splits, math.ceil(target / blocks))

    def waste(parallel_units: int) -> float:
        if parallel_units <= 0:
            return 1.0
        waves = math.ceil(parallel_units / target)
        return 1.0 - parallel_units / (waves * target)

    return {
        "total_blocks": blocks,
        "target_blocks": target,
        "splits": splits,
        "waste_ratio_before": round(waste(blocks), 4),
        "waste_ratio_after": round(waste(blocks * splits), 4),
    }


def efficiency(measured_ms: float, lower_bound_ms: float) -> float:
    """实测 / 理论下界 > 1 表示还有优化空间；越接近 1 越好。"""
    return measured_ms / lower_bound_ms


@dataclass(frozen=True)
class ModelSize:
    """只保留 roofline 需要的几个量。"""

    name: str
    params: float
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2

    @property
    def kv_per_token(self) -> int:
        return kv_bytes_per_token(self.layers, self.kv_heads, self.head_dim, self.dtype_bytes)

    @property
    def weight_bytes(self) -> int:
        return weights_bytes(self.params, self.dtype_bytes)


LLAMA3_8B = ModelSize(
    name="Llama-3-8B",
    params=8.03e9,
    layers=32,
    kv_heads=8,
    head_dim=128,
)
