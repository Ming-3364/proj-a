# ADR 0002：硬件基线（待 W0 实测填写）

日期：2026-11-18 ｜ 状态：**已填**（实测完成）

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
GPU 名称            : NVIDIA H100 PCIe
compute capability  : 9.0 (sm_90)
SM 数               : 114          <- 与数据表一致
smem / SM           : 228 KB       <- 与数据表一致
smem / CTA (opt-in) : 227 KB
L2                  : 50 MB
显存                : 84.93 GB (≈79 GiB，80GB 型号的可用值)
驱动 / CUDA         : 12.4
实测 copy 带宽      : 1.70 TB/s    <- 数据表 2.0；达成率 85%
实测 read 带宽      : 1.94 TB/s    <- 数据表 2.0；达成率 97%
```

**采用的建模分母**：`BW = 1.94 TB/s (read)`。
理由：decode 以读为主（权重 + KV cache），copy 的读+写混合不代表这个场景。

**影响**：bs=1 / ctx=4K 的 decode 下界由 8.30 ms（按 2.0）变为 **8.55 ms**（按 1.94），
偏差 3%——比 SXM 的 3.35 TB/s 造成的 1.7× 误差小得多，但仍必须用实测值。

## 代码里的约定

- 所有性能公式只接受 `BW_MEASURED`（常量从本 ADR 读入），不接受硬编码数据表值；
- 报告中的「带宽达成率」分子是实测有效带宽，分母是**本 ADR 的实测峰值**；
- 若实测与数据表偏差 > 15%，在报告里单独说明（可能的原因：功耗/时钟设置、NUMA、MIG 等）。


## 多设备策略（H100 / A100 切换）

实验会在 **H100 PCIe (sm_90)** 与 **A100 (sm_80)** 之间切换。不同卡不能共用产物：

| 项 | H100 PCIe | A100 80G | 是否影响产物 |
|---|---|---|---|
| arch | `sm_90` | `sm_80` | **是**：cubin 与架构绑定，PTX 才向上兼容 |
| smem / SM | 228 KB | 164 KB | **是**：改变合法 tile 集合 |
| SM 数 | 114 | 108 | 是：wave quantization / split 启发式 |
| L2 | 50 MB | 40 MB | 是：L2 tile 复用半径 |
| HBM | 实测 1.94 TB/s | 待实测（数据表 2.04） | 是：roofline 分母 |
| MMA | `wgmma` | `mma.sync` | **是**：lowering 的 intrinsic 分叉 |
| 异步拷贝 | TMA (`cp.async.bulk`) | `cp.async` | 是：pipelining pass 分叉 |

**约定**：

1. 单一事实源是 `analysis/hardware.py` 的 `DeviceSpec`（`resolve()` / `detect()`）。
2. 所有 arch 相关参数只从 `DeviceSpec.arch` 取，**禁止硬编码 `"sm_90"`**。
3. 产物按架构分目录：`build/sm_90/`、`build/sm_80/`。
4. 结果 JSON 按设备分目录：`benchmarks/results/<device-name>/`。
5. 切换卡的复验命令：`python scripts/gpu_hello.py`（自动识别）或 `--arch sm_80`。

**A100 待办**：跑一次 `python bench/bw_probe.py`，把实测带宽填进 `MEASURED_TBPS`。
