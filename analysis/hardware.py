"""硬件档案（DeviceProfile）：多卡切换的唯一事实源。

为什么需要它：H100 与 A100 不是"换一台机器"，而是**换一套硬件语义**。

| | H100 PCIe | A100 80G | 影响的层 |
|---|---|---|---|
| 架构 | sm_90 | sm_80 | `llc -mcpu` / `ptxas -arch` / cubin 兼容性 |
| SM 数 | 114 | 108 | wave quantization、并行度上限 |
| smem / SM | 228 KB | 164 KB | tiling 的 smem 预算（直接改变合法 tile 集合） |
| L2 | 50 MB | 40 MB | L2 tile 的复用半径 |
| HBM | ~2.0 TB/s（实测 1.94） | ~2.0 TB/s（80G）/ 1.55（40G） | roofline 分母 |
| MMA | `wgmma`(sm_90a) | `mma.sync` | lowering 的 intrinsic 选择 |
| 异步拷贝 | `cp.async.bulk`(TMA) | `cp.async` | pipelining pass |

**铁律**：cubin 与架构绑定（sm_90 的 cubin 不能在 sm_80 上跑）；PTX 才向上兼容。
所以按设备分别构建，产物按 arch 分目录。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = [
    "DeviceSpec",
    "PROFILES",
    "H100_PCIE",
    "H100_SXM",
    "A100_80G",
    "A100_40G",
    "arch_from_cc",
    "resolve",
    "detect",
    "MEASURED_TBPS",
    "MEASURED_COPY_TBPS",
    "MEASURED_READ_TBPS",
    "H100_PCIE_MEASURED",
]

# 特性开关：写成集合而不是布尔字段，方便 lowering 里做能力判断
FEAT_MMA_SYNC = "mma_sync"
FEAT_CP_ASYNC = "cp_async"
FEAT_WGMMA = "wgmma"
FEAT_TMA = "tma"


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    arch: str  # 例如 "sm_90" / "sm_80"，直接喂给 llc -mcpu 与 ptxas -arch
    bandwidth_tbps: float  # 峰值 HBM（TB/s）
    sm_count: int
    smem_per_sm_kb: int = 228
    smem_per_cta_max_kb: int = 227
    l2_mb: int = 50
    max_warps_per_sm: int = 64
    max_ctas_per_sm: int = 32
    tensor_tflops_fp16: float = 0.0
    features: frozenset[str] = frozenset()

    @property
    def bandwidth_bps(self) -> float:
        return self.bandwidth_tbps * 1e12

    @property
    def cc(self) -> tuple[int, int]:
        """sm_90 -> (9, 0)"""
        digits = self.arch.removeprefix("sm_")
        return int(digits[0]), int(digits[1]) if len(digits) > 1 else 0

    def has(self, feature: str) -> bool:
        return feature in self.features

    def with_measured_bandwidth(self, tbps: float) -> "DeviceSpec":
        return replace(self, bandwidth_tbps=tbps)


def arch_from_cc(major: int, minor: int) -> str:
    return f"sm_{major}{minor}"


H100_PCIE = DeviceSpec(
    name="H100 PCIe 80GB",
    arch="sm_90",
    bandwidth_tbps=2.00,  # 实测 1.94（见 ADR 0002）
    sm_count=114,
    tensor_tflops_fp16=756.0,
    features=frozenset({FEAT_MMA_SYNC, FEAT_CP_ASYNC, FEAT_WGMMA, FEAT_TMA}),
)

H100_SXM = DeviceSpec(
    name="H100 SXM 80GB",
    arch="sm_90",
    bandwidth_tbps=3.35,
    sm_count=132,
    tensor_tflops_fp16=989.0,
    features=frozenset({FEAT_MMA_SYNC, FEAT_CP_ASYNC, FEAT_WGMMA, FEAT_TMA}),
)

A100_80G = DeviceSpec(
    name="A100 80GB",
    arch="sm_80",
    bandwidth_tbps=2.04,  # 数据表 2039 GB/s；**必须用 bw_probe 实测校准**
    sm_count=108,
    smem_per_sm_kb=164,
    smem_per_cta_max_kb=163,
    l2_mb=40,
    tensor_tflops_fp16=312.0,
    features=frozenset({FEAT_MMA_SYNC, FEAT_CP_ASYNC}),
)

A100_40G = DeviceSpec(
    name="A100 40GB",
    arch="sm_80",
    bandwidth_tbps=1.55,
    sm_count=108,
    smem_per_sm_kb=164,
    smem_per_cta_max_kb=163,
    l2_mb=40,
    tensor_tflops_fp16=312.0,
    features=frozenset({FEAT_MMA_SYNC, FEAT_CP_ASYNC}),
)

PROFILES: dict[str, DeviceSpec] = {
    "h100-pcie": H100_PCIE,
    "h100-sxm": H100_SXM,
    "a100-80g": A100_80G,
    "a100-40g": A100_40G,
}

# ---- 实测值（W0：H100 PCIe 已测；A100 待测）----
MEASURED_COPY_TBPS = 1.70
MEASURED_READ_TBPS = 1.94
MEASURED_TBPS: dict[str, float] = {H100_PCIE.name: MEASURED_READ_TBPS}

H100_PCIE_MEASURED = H100_PCIE.with_measured_bandwidth(MEASURED_READ_TBPS)


def resolve(name: str | None) -> DeviceSpec:
    """按名字取档案；None 或 'auto' 时自动探测（无 GPU 则回落到 H100 PCIe）。"""
    if not name or name == "auto":
        return detect() or H100_PCIE
    key = name.lower()
    if key in PROFILES:
        return PROFILES[key]
    for spec in PROFILES.values():
        if spec.arch == key or spec.name.lower() == key:
            return spec
    raise ValueError(f"未知设备 {name!r}；可选：{', '.join(PROFILES)}")


def detect() -> DeviceSpec | None:
    """在 GPU 机器上按实际设备挑档案（torch 不可用时返回 None）。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        p = torch.cuda.get_device_properties(0)
    except Exception:
        return None

    arch = arch_from_cc(p.major, p.minor)
    for spec in PROFILES.values():
        if spec.arch == arch and spec.sm_count == p.multi_processor_count:
            return spec
    for spec in PROFILES.values():
        if spec.arch == arch:
            return spec
    # 未知型号：用实测到的属性临时造一个（宁可跑起来，也不要硬编码失败）
    return DeviceSpec(
        name=p.name,
        arch=arch,
        bandwidth_tbps=0.0,  # 0 表示"未标定"，roofline 会显式报错
        sm_count=p.multi_processor_count,
        features=frozenset({FEAT_MMA_SYNC, FEAT_CP_ASYNC}),
    )
