"""Shared memory bank conflict 模型（纯 Python，Mac 上即可验证）。

⚠️ 必须先讲清的事实：**Triton 不暴露 shared memory 布局。**
你无法像 CUDA 那样写 `__shared__ float s[32][33]` 来 padding 消冲突——
smem 的 padding 与 XOR swizzle 由后端在 TTGIR 的
`#ttg.swizzled_shared<{vec, perPhase, maxPhase, order}>` 里决定。

所以本模块的定位是「**预测器**」：给定一个布局假设，算出访问模式会撞多少 bank，
再用 ncu 的 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_*` 去校准它。

硬件模型（H100）：
  - 32 个 bank，每个 bank 宽 4 byte；bank = (byte_addr // 4) % 32
  - 同一 wavefront 内，落在**同一个 bank 的不同 4B word** 上 = 冲突
  - 落在同一个 word（或不同 bank）不算冲突（广播是免费的）
  - 2 字节元素（fp16/bf16）：相邻两个元素共享一个 bank word
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "Layout",
    "linear",
    "padded",
    "xor_swizzled",
    "access_rows",
    "access_cols",
    "conflicts",
    "report",
    "BANKS",
    "BANK_WIDTH",
]

BANKS = 32
BANK_WIDTH = 4  # bytes


@dataclass(frozen=True)
class Layout:
    """一个 row-major tile 的物理布局假设。"""

    rows: int
    cols: int
    dtype_bytes: int = 4
    row_stride_elems: int | None = None  # None -> 无 padding
    # XOR swizzle：以 vec 个元素为一个 chunk，按 (row % max_phase) 做异或
    vec: int = 0  # 0 表示不 swizzle
    max_phase: int = 1

    @property
    def stride(self) -> int:
        return self.cols if self.row_stride_elems is None else self.row_stride_elems

    def word_addr(self, r: int, c: int) -> int:
        """逻辑坐标 (r, c) 对应的 4B word 地址。"""
        if self.vec > 0:
            chunk, within = divmod(c, self.vec)
            chunk ^= r % self.max_phase
            c = chunk * self.vec + within
        return (r * self.stride + c) * self.dtype_bytes // BANK_WIDTH

    def bank(self, r: int, c: int) -> int:
        return self.word_addr(r, c) % BANKS


def linear(rows: int, cols: int, dtype_bytes: int = 4) -> Layout:
    return Layout(rows, cols, dtype_bytes)


def padded(rows: int, cols: int, pad: int = 1, dtype_bytes: int = 4) -> Layout:
    """CUDA 里最常见的手法：row stride 加 pad，让列访问落在不同 bank。"""
    return Layout(rows, cols, dtype_bytes, row_stride_elems=cols + pad)


def xor_swizzled(
    rows: int, cols: int, dtype_bytes: int = 4, vec: int = 4, max_phase: int = 8
) -> Layout:
    """Triton/CUTLASS 风格：按 128B chunk 做 XOR swizzle。

    vec 取 128B / dtype_bytes（fp32 -> 32? 实际常用 vec = 128B 内元素数 / …），
    本模型把它当参数，具体取值以 TTGIR dump 出来的编码为准。
    """
    return Layout(rows, cols, dtype_bytes, vec=vec, max_phase=max_phase)


def access_rows(r: int, cols: int) -> list[tuple[int, int]]:
    """一个 warp 连续读同一行的 cols 个元素（coalesced 的典型形态）。"""
    return [(r, c) for c in range(cols)]


def access_cols(c: int, rows: int) -> list[tuple[int, int]]:
    """一个 warp 逐行读同一列（转置访问，冲突重灾区）。"""
    return [(r, c) for r in range(rows)]


def conflicts(layout: Layout, accesses: list[tuple[int, int]]) -> dict:
    """统计一次访问（不超过 32 lane）的 bank 冲突。

    返回：
      banks_hit         : 涉及的 bank 数
      max_multiplicity  : 单个 bank 上最多有几次「不同 word」的访问
      extra_wavefronts  : 需要额外多少个 wavefront（= max_multiplicity - 1）
      word_conflict     : max_multiplicity > 1 即为有冲突
    """
    per_bank: dict[int, set[int]] = {}
    for r, c in accesses:
        b = layout.bank(r, c)
        per_bank.setdefault(b, set()).add(layout.word_addr(r, c))

    mult = [len(v) for v in per_bank.values()]
    worst = max(mult) if mult else 1
    return {
        "lanes": len(accesses),
        "banks_hit": len(per_bank),
        "max_multiplicity": worst,
        "extra_wavefronts": worst - 1,
        "word_conflict": worst > 1,
    }


def report(layout: Layout, pattern: str = "col") -> dict:
    """对一个 tile 扫一遍各 lane 的访问，给出冲突汇总（便于和 ncu 对照）。"""
    if pattern == "col":
        groups = [access_cols(c, layout.rows) for c in range(layout.cols)]
    elif pattern == "row":
        groups = [access_rows(r, layout.cols) for r in range(layout.rows)]
    else:
        raise ValueError(pattern)

    extras = [conflicts(layout, g)["extra_wavefronts"] for g in groups]
    return {
        "layout": f"{layout.rows}x{layout.cols} dtype={layout.dtype_bytes*8}bit "
        f"stride={layout.stride} vec={layout.vec} maxPhase={layout.max_phase}",
        "pattern": pattern,
        "mean_extra_wavefronts": round(sum(extras) / len(extras), 2),
        "max_extra_wavefronts": max(extras),
        "conflict_free": all(e == 0 for e in extras),
    }
