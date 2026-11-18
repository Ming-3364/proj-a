"""SMEM 预算 → occupancy 上界。纯 Python，Mac 上即可验证。

这是「L1 tiling」的定量部分：tile 尺寸不是一个美学选择，
它先被 smem 预算卡住，再被 occupancy 卡住。

核心公式（要能白板推）：

    smem_per_stage = (BLOCK_M·BLOCK_K + BLOCK_K·BLOCK_N) · dtype_bytes
    smem_total     = smem_per_stage · num_stages          (+ epilogue 缓冲)

例：128×128×32, fp16, 3 stages → (128·32 + 32·128)·2·3 = 49,152 B = 48 KiB
"""

from __future__ import annotations

from dataclasses import dataclass

from analysis.hardware import H100_PCIE, DeviceSpec

__all__ = ["TileSpec", "smem_bytes", "ctas_per_sm", "occupancy"]


@dataclass(frozen=True)
class TileSpec:
    block_m: int
    block_n: int
    block_k: int
    dtype_bytes: int = 2  # fp16/bf16
    num_stages: int = 3
    num_warps: int = 4
    extra_smem_bytes: int = 0  # epilogue / reduction 缓冲

    def __str__(self) -> str:
        return (
            f"{self.block_m}x{self.block_n}x{self.block_k} "
            f"{self.dtype_bytes*8}bit stages={self.num_stages} "
            f"warps={self.num_warps}"
        )


def smem_bytes(t: TileSpec) -> int:
    """一个 CTA 的 shared memory 需求（字节）。"""
    per_stage = (t.block_m * t.block_k + t.block_k * t.block_n) * t.dtype_bytes
    return per_stage * t.num_stages + t.extra_smem_bytes


def ctas_per_sm(t: TileSpec, dev: DeviceSpec = H100_PCIE) -> int:
    """受 smem 限制时，每个 SM 最多驻留几个 CTA。

    注意单 CTA 上限（H100 为 227KB，需要 opt-in 超大动态 smem）：
    超过它的 tile 根本启动不了。
    """
    need = smem_bytes(t)
    if need > dev.smem_per_cta_max_kb * 1024:
        return 0  # 不可启动
    by_smem = (dev.smem_per_sm_kb * 1024) // need
    return max(0, min(dev.max_ctas_per_sm, int(by_smem)))


def occupancy(t: TileSpec, dev: DeviceSpec = H100_PCIE) -> dict:
    """返回 occupancy 相关的一整套数（warps 口径）。"""
    n_cta = ctas_per_sm(t, dev)
    warps = n_cta * t.num_warps
    return {
        "tile": str(t),
        "smem_bytes": smem_bytes(t),
        "smem_kb": round(smem_bytes(t) / 1024, 2),
        "launchable": n_cta > 0,
        "ctas_per_sm": n_cta,
        "warps_per_sm": warps,
        "max_warps_per_sm": dev.max_warps_per_sm,
        "occupancy_pct": round(100.0 * warps / dev.max_warps_per_sm, 1),
    }


def sweep(
    block_m=128, block_n=128, block_k_candidates=(16, 32, 64, 128),
    stages_candidates=(2, 3, 4, 5), dev: DeviceSpec = H100_PCIE,
) -> list[dict]:
    """扫一遍 (BLOCK_K, num_stages)，看谁先把 smem/occupancy 打死。"""
    out = []
    for bk in block_k_candidates:
        for st in stages_candidates:
            t = TileSpec(block_m, block_n, bk, num_stages=st)
            row = occupancy(t, dev)
            row["block_k"], row["num_stages"] = bk, st
            out.append(row)
    return out
