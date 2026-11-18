"""分析模型（smem/occupancy、bank、roofline）的测试。

这些是纯数学断言：不需要 GPU、不需要 triton，在 Mac 上随时可跑。
它们的价值在于「把计划文档里的数字钉死在代码里」——文档会漂移，测试不会。
"""

from __future__ import annotations

from analysis import bank_model as bm
from analysis import roofline as rl
from analysis.hardware import H100_PCIE, H100_SXM
from analysis.smem_occupancy import TileSpec, ctas_per_sm, occupancy, smem_bytes


# ---------------------------------------------------------------- smem / occupancy

def test_smem_formula_matches_hand_calc():
    """128x128x32 fp16, 3 stages -> (128*32 + 32*128)*2*3 = 49,152 B = 48 KiB。"""
    t = TileSpec(128, 128, 32, dtype_bytes=2, num_stages=3)
    assert smem_bytes(t) == 49152, smem_bytes(t)


def test_occupancy_for_48kb_tile():
    """228 KiB / 48 KiB = 4.75 -> 4 个 CTA；4 warps 每个 -> 16/64 warps = 25%。"""
    o = occupancy(TileSpec(128, 128, 32, num_stages=3, num_warps=4))
    assert o["ctas_per_sm"] == 4, o
    assert o["warps_per_sm"] == 16
    assert o["occupancy_pct"] == 25.0


def test_bigger_tile_reduces_resident_ctas():
    """128x128x64 3-stage = 96 KiB -> 2 CTA；再加 stage 就更少。"""
    assert ctas_per_sm(TileSpec(128, 128, 64, num_stages=3)) == 2
    assert ctas_per_sm(TileSpec(128, 128, 64, num_stages=5)) == 1


def test_oversized_tile_is_not_launchable():
    """128x256x128 4-stage = (128*128+128*256)*2*4 = 393,216 B > 227 KiB 单 CTA 上限。"""
    t = TileSpec(128, 256, 128, num_stages=4)
    assert smem_bytes(t) == 393216
    assert ctas_per_sm(t) == 0, "超过单 CTA 上限时应当判定为不可启动"
    assert occupancy(t)["launchable"] is False


def test_sxm_has_same_smem_budget():
    """PCIe 与 SXM 的 smem 一致，所以 tile 预算结论可以复用（带宽不能复用）。"""
    t = TileSpec(128, 128, 32, num_stages=3)
    assert ctas_per_sm(t, H100_PCIE) == ctas_per_sm(t, H100_SXM)


# ---------------------------------------------------------------- bank conflict

def test_linear_layout_column_access_is_32way():
    """行 stride=32 个 fp32 时，同列 32 次访问全落在同一个 bank -> 31 额外 wavefront。"""
    r = bm.report(bm.linear(32, 32, 4), "col")
    assert r["max_extra_wavefronts"] == 31, r
    assert r["conflict_free"] is False


def test_fp32_padding_removes_column_conflict():
    """经典手法：stride 33 -> (r*33 + c) % 32 = (r+c) % 32，32 行各占一个 bank。"""
    assert bm.report(bm.padded(32, 32, pad=1, dtype_bytes=4), "col")["conflict_free"] is True


def test_fp16_padding_must_shift_a_full_bank_word():
    """⚠️ 容易被忽略的陷阱：fp16 下 pad=1 只偏移 2 字节，换不了 4B bank word。

    pad=0 -> 31 额外；pad=1 -> 仍有 1；**pad=2 才 conflict-free**（跳过一个完整 word）。
    """
    assert bm.report(bm.linear(32, 64, 2), "col")["max_extra_wavefronts"] == 31
    assert bm.report(bm.padded(32, 64, pad=1, dtype_bytes=2), "col")["max_extra_wavefronts"] == 1
    assert bm.report(bm.padded(32, 64, pad=2, dtype_bytes=2), "col")["conflict_free"] is True


def test_row_access_is_conflict_free_regardless_of_layout():
    """连续读同一行是 coalesced 的典型形态，任何布局都不该有 bank 冲突。"""
    for layout in (bm.linear(64, 32, 4), bm.linear(64, 32, 2),
                   bm.xor_swizzled(64, 32, 2, vec=8, max_phase=8)):
        assert bm.report(layout, "row")["conflict_free"] is True


def test_swizzle_is_not_a_magic_fix_for_transpose():
    """XOR swizzle 是为 dot operand / ldmatrix 设计的，不能指望它解决任意转置访问。

    这里刻意断言「仍然有冲突」：如果哪天模型说 0 冲突，说明模型写错了。
    """
    r = bm.report(bm.xor_swizzled(32, 32, 4, vec=8, max_phase=8), "col")
    assert r["max_extra_wavefronts"] > 0
    assert r["max_extra_wavefronts"] < 31, "swizzle 应当显著缓解，但不是归零"


# ---------------------------------------------------------------- roofline

def test_kv_bytes_per_token():
    """32 层 · 2(K,V) · 8 KV head · 128 dim · 2 B = 131,072 B = 128 KiB。"""
    assert rl.kv_bytes_per_token(32, 8, 128, 2) == 131072
    assert rl.LLAMA3_8B.kv_per_token == 131072


def test_decode_lower_bound_on_h100_pcie():
    """bs=1 / 4K：权重 16.06 GB + KV 0.54 GB，2.0 TB/s -> ~8.3 ms。"""
    m = rl.LLAMA3_8B
    lb = rl.decode_lower_bound_ms(1, 4096, m.weight_bytes, m.kv_per_token, H100_PCIE)
    assert 8.0 < lb < 8.6, lb

    # 同一个 workload 在 SXM 上快 1.7×：这就是「别照抄网上数字」的原因
    lb_sxm = rl.decode_lower_bound_ms(1, 4096, m.weight_bytes, m.kv_per_token, H100_SXM)
    assert lb_sxm < lb


def test_crossover_batch_is_bandwidth_independent():
    """交叉点只由 权重/KV 决定，换硬件不变：4K -> ~30，8K -> ~15。"""
    m = rl.LLAMA3_8B
    c4 = rl.crossover_batch(m.weight_bytes, 4096, m.kv_per_token)
    c8 = rl.crossover_batch(m.weight_bytes, 8192, m.kv_per_token)
    assert 29 < c4 < 31, c4
    assert 14 < c8 < 16, c8
    assert abs(c4 / 2 - c8) < 0.2, "上下文翻倍，交叉点应减半（c4/2 ≈ c8）"


def test_split_kv_recovers_idle_parallelism():
    """bs=1, ctx=4K, BLOCK_N=128 -> 仅 32 个 block，而 114 个 SM 槽位 -> 需要 split。"""
    r = rl.split_kv_blocks(batch=1, seq_len=4096, block_n=128, dev=H100_PCIE, ctas_per_sm=1)
    assert r["total_blocks"] == 32
    assert r["target_blocks"] == 114
    assert r["splits"] == 4, r
    assert r["waste_ratio_after"] < r["waste_ratio_before"], "split 应当降低闲置率"


def test_split_kv_not_needed_when_already_enough_blocks():
    """bs=32 时已经有 1024 个 block，不需要 split。"""
    r = rl.split_kv_blocks(batch=32, seq_len=4096, block_n=128, dev=H100_PCIE, ctas_per_sm=1)
    assert r["total_blocks"] == 1024
    assert r["splits"] == 1
