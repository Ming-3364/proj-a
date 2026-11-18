# ADR 0002：硬件基线（待 W0 实测填写）

日期：2026-11-11 ｜ 状态：**待填**（W0 的硬性产出）

## 为什么必须实测

同一个型号名下有多个硬件版本，带宽与 SM 数差异巨大。以 H100 为例：

| 版本 | HBM | 峰值带宽 | SM 数 |
|---|---|---|---|
| H100 SXM | 80 GB HBM3 | ≈3.35 TB/s | 132 |
| **H100 PCIe** | 80 GB HBM2e | **≈2.0 TB/s** | **114** |
| H100 NVL | 94 GB HBM3 | ≈3.9 TB/s | 132 |

本项目 GPU 是 **H100 PCIe**。若照抄 SXM 数字，所有 roofline 结论会偏移约 1.7×，
并行度/wave quantization 分析也会错位。

## 待填项（跑 `make bw` 后替换）

```
GPU 名称            : <nvidia-smi 输出>
compute capability  : <期望 9.0>
SM 数               : <期望 114>
smem / SM           : <期望 228 KB>
L2                  : <期望 50 MB>
实测 copy 带宽      : ____ TB/s   <- bw_probe
实测 read 带宽      : ____ TB/s   <- bw_probe
实测结果文件        : benchmarks/results/<gpu>/bw_probe_<sha>.json
```

## 代码里的约定

- 所有性能公式只接受 `BW_MEASURED`（常量从本 ADR 读入），不接受硬编码数据表值；
- 报告中的「带宽达成率」分子是实测有效带宽，分母是**本 ADR 的实测峰值**；
- 若实测与数据表偏差 > 15%，在报告里单独说明（可能的原因：功耗/时钟设置、NUMA、MIG 等）。
