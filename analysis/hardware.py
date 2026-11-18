"""硬件参数表 + 单一定义源。

⚠️ 核心约定：**带宽一律用 BW_MEASURED（实测），不要用数据表。**

H100 有三个版本，网上的教程几乎都默认 SXM，直接照抄会让所有 roofline 结论偏移 1.7×：

| 版本 | HBM | 峰值带宽 | SM 数 |
|---|---|---|---|
| H100 SXM   | 80 GB HBM3  | 3.35 TB/s | 132 |
| H100 PCIe  | 80 GB HBM2e | 2.00 TB/s | 114 |  <- 本项目
| H100 NVL   | 94 GB HBM3  | 3.90 TB/s | 132 |

W0 跑 `bench/bw_probe.py` 后，把实测值填进 `docs/adr/0002-hw-baseline.md`，
并通过 `set_measured_bandwidth()` 注入本模块（避免任何代码里硬编码数据表值）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = [
    "DeviceSpec",
    "H100_PCIE",
    "H100_SXM",
    "MEASURED_TBPS",
    "MEASURED_COPY_TBPS",
    "MEASURED_READ_TBPS",
    "H100_PCIE_MEASURED",
    "set_measured_bandwidth",
]


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    bandwidth_tbps: float  # 峰值 HBM 带宽（TB/s = 1e12 B/s）
    sm_count: int
    smem_per_sm_kb: int = 228  # H100 全系一致
    smem_per_cta_max_kb: int = 227  # opt-in 后单 CTA 上限
    l2_mb: int = 50
    max_warps_per_sm: int = 64
    max_ctas_per_sm: int = 32
    tensor_tflops_fp16: float = 0.0  # 仅用于 roofline 对照，缺省 0 表示未知

    @property
    def bandwidth_bps(self) -> float:
        return self.bandwidth_tbps * 1e12

    def with_measured_bandwidth(self, tbps: float) -> "DeviceSpec":
        return replace(self, bandwidth_tbps=tbps)


# 数据表值：只用于「量级参考」，不用于结论
H100_PCIE = DeviceSpec(
    name="H100 PCIe 80GB",
    bandwidth_tbps=2.00,
    sm_count=114,
    tensor_tflops_fp16=756.0,
)

H100_SXM = DeviceSpec(
    name="H100 SXM 80GB",
    bandwidth_tbps=3.35,
    sm_count=132,
    tensor_tflops_fp16=989.0,
)

# ---- W0 实测（2026-11-18，H100 PCIe / driver 12.4，bench/bw_probe.py）----
# copy 1.70 TB/s（读+写）/ read 1.94 TB/s（纯读）
# decode 以读为主（权重 + KV），所以分母用 read。
MEASURED_COPY_TBPS = 1.70
MEASURED_READ_TBPS = 1.94

# 结论性常量：默认就用实测值（数据表 2.0 只作参考）
H100_PCIE_MEASURED = H100_PCIE.with_measured_bandwidth(MEASURED_READ_TBPS)

MEASURED_TBPS: dict[str, float] = {H100_PCIE.name: MEASURED_READ_TBPS}


def set_measured_bandwidth(tbps: float, name: str = H100_PCIE.name) -> DeviceSpec:
    """注入实测带宽，返回校准后的 spec。"""
    if not (0 < tbps < 100):
        raise ValueError(f"带宽看起来不像 TB/s: {tbps}")
    MEASURED_TBPS[name] = tbps
    return H100_PCIE.with_measured_bandwidth(tbps)
