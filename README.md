# minitriton — 自研迷你 Triton 编译器

**目标**：吃一段 Triton 子集源码，走 **TTIR → TTGIR → LLVM IR → PTX → cubin**，
输出能在 GPU 上跑的 kernel；**tiling 与 shared memory 布局（bank conflict）由编译器 pass 推导**，不是手写的。
官方 Triton 只作对照基准，不是依赖。

判据：如果 `BLOCK_M` 和 swizzle 是人手填的，那是在写 kernel；由 pass 从 IR 推出来，才是编译器。

**路线图：[PLAN.md](PLAN.md)**
上一版（kernel lab）计划：`docs/archive/plan-kernel-lab-v1.md`

## 流水线

```
Triton 子集 → frontend → TTIR → [P1 tiling] → TTGIR → [P2 layout] → [P3 流水]
                                        → LLVM IR → llc(NVPTX) → PTX → ptxas → cubin → GPU
```

三个决定：不上 MLIR 但照抄分层（TTIR/TTGIR 边界）、后端落 LLVM IR（不是 CUDA C）、
MMA 先标量循环、W3 再试 NVVM `mma.sync` intrinsic。

## 环境

| | |
|---|---|
| 本地 | Apple Silicon，无 GPU；**macOS 装不了 triton**（wheel 里是 AArch64 ELF） |
| 本地跑 Triton | 仅 Linux/arm64 容器 + `TRITON_INTERPRET=1`（验正确性，不验性能） |
| GPU 机 | H100 PCIe 80GB（sm_90a，≈2.0 TB/s，114 SM），ncu 可用 |

编译器本身**大部分不需要 GPU**：IR、pass、codegen 都能在 Mac 上做 golden 快照测试；只有 nvcc 编译、跑起来、ncu 必须上卡。

## 命令

```bash
make doctor      # 体检解释器（PATH 里可能有多个 python3）
make test        # 本地纯 CPU：分析模型 + 参考实现
make dev-shell   # Linux/arm64 容器（triton 3.9.0 + torch 2.9.1）
make bw          # 实测带宽（需 GPU）
```

## 在 GPU 机器上运行（H100 PCIe）

```bash
git clone https://github.com/Ming-3364/proj-a.git && cd proj-a

# 1) Python 依赖（本地模型只要 numpy；对拍才需要 torch）
python3 -m venv .venv && . .venv/bin/activate
pip install numpy pytest

# 2) 工具链：llc 必须带 NVPTX 后端，ptxas 用来出 cubin
sudo apt-get install -y llvm          # Debian/Ubuntu；已有 CUDA toolkit 则 ptxas 在 $CUDA_HOME/bin
pip install triton==3.9.0             # 自带 ptxas，免装 CUDA toolkit（不要覆盖 torch 依赖的那份）

# 3) 依次验证
python -m minitriton.runtime.launch   # 驱动/设备探测（sm_90? 驱动版本?）
make smoke-cubin                      # .ll → .ptx → .cubin（纯 CPU）
python scripts/gpu_hello.py --n 1024  # S1 端到端：上卡执行 + 对拍 → 期望 PASS
python bench/bw_probe.py              # 顺便标定 HBM 带宽，写进 docs/adr/0002
```

**期望输出**（`gpu_hello.py`）：

```
[hello] 设备：H100 PCIe | sm_90 | driver 12.x | 79.2 GB
[hello] grid=(1,1,1) block=(1024,1,1) n=1024
[hello] 前 8 个输出: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
[hello] PASS  耗时 ... µs
```

**如果失败，优先看这三处**（按概率排序）：

1. `cuModuleLoadData` 失败 → PTX 版本高于驱动支持（本地 LLVM 出 `.version 7.8`，需 CUDA 12.0+ 驱动）。
2. `cuModuleGetFunction(hello)` 失败 → cubin 里没有 `.entry hello`（说明 `.ll` 少了 `ptx_kernel` 调用约定）。
3. launch 崩溃/结果全 0 → `kernelParams` 编排错（`CUdeviceptr` 必须是 64 位，顺序必须与签名一致）。

## 当前状态

| 项 | 状态 |
|---|---|
| S1 前半：`.ll → .ptx → .cubin`（Mac 容器内） | ✅ 已完成，纳入回归（28 passed） |
| S1 后半：`libcuda` launcher + 上卡执行 | ⚠️ 代码已写，**待首次上卡验证** |
| 分析模型 / 参考实现 / 带宽标定 | ✅ 已有 |
| S2（TTIR→LLVM）/ S3（前端）/ S4-S6 | ⬜ 未开始 |

## 目录

```
minitriton/{frontend,ir,passes,codegen,runtime}/   # 编译器本体（逐步落地）
analysis/      # 编译器的代价模型：smem/occupancy、bank、roofline
refs/          # torch 参考实现（正确性裁判）
kernels/       # 手写基线（对照用）
bench/         # 带宽标定 + 与官方对照
tests/         # 零依赖 runner + golden 快照
docs/adr/      # 决策记录（范围、硬件基线）
```
