"""设备档案（H100 / A100 切换）的测试。

这些断言守住的是"换卡时要改哪些地方"的契约：
arch 决定 llc/ptxas 参数与 cubin 兼容性；smem/SM 数决定 tiling 的合法集；
features 决定 lowering 走 wgmma/TMA 还是 mma.sync/cp.async。
"""

from __future__ import annotations

from analysis import hardware as hw
from analysis.smem_occupancy import TileSpec, ctas_per_sm


def test_h100_is_sm90_with_hopper_features():
    d = hw.H100_PCIE
    assert d.arch == "sm_90"
    assert d.cc == (9, 0)
    assert d.sm_count == 114
    assert d.smem_per_sm_kb == 228
    assert d.has(hw.FEAT_WGMMA) and d.has(hw.FEAT_TMA)


def test_a100_is_sm80_without_hopper_features():
    """A100 没有 wgmma、没有 TMA —— 这是 lowering 必须分叉的根本原因。"""
    d = hw.A100_80G
    assert d.arch == "sm_80"
    assert d.cc == (8, 0)
    assert d.sm_count == 108
    assert d.smem_per_sm_kb == 164
    assert d.smem_per_cta_max_kb == 163
    assert not d.has(hw.FEAT_WGMMA)
    assert not d.has(hw.FEAT_TMA)
    assert d.has(hw.FEAT_MMA_SYNC) and d.has(hw.FEAT_CP_ASYNC)


def test_smem_budget_changes_the_legal_tile_set():
    """同一块 tile 在 A100 上驻留的 CTA 更少 —— 所以 tiling 不能写死 H100 的数字。"""
    t = TileSpec(128, 128, 64, num_stages=3)  # 96 KiB
    assert ctas_per_sm(t, hw.H100_PCIE) == 2
    assert ctas_per_sm(t, hw.A100_80G) == 1  # 164KB // 96KB = 1

    # 128x128x64 5-stage = 160 KiB：H100 能跑，A100 超过单 CTA 上限（163KB 边缘）
    big = TileSpec(128, 128, 64, num_stages=6)  # 192 KiB
    assert ctas_per_sm(big, hw.H100_PCIE) == 1
    assert ctas_per_sm(big, hw.A100_80G) == 0, "A100 单 CTA 上限 163KB，应判不可启动"


def test_arch_from_cc_and_resolve():
    assert hw.arch_from_cc(9, 0) == "sm_90"
    assert hw.arch_from_cc(8, 0) == "sm_80"
    assert hw.resolve("h100-pcie") is hw.H100_PCIE
    assert hw.resolve("a100-80g") is hw.A100_80G
    assert hw.resolve("sm_80") is hw.A100_80G  # 也接受架构名
    assert hw.resolve(None) is hw.H100_PCIE  # 无 GPU 时的回落


def test_unknown_profile_raises_with_choices():
    try:
        hw.resolve("v100")
    except ValueError as e:
        assert "v100" in str(e)
        assert "h100-pcie" in str(e), "报错应列出可选档案"
    else:
        raise AssertionError("应当抛 ValueError")
