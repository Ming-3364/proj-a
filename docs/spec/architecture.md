# minitriton 架构：复刻 Triton 的 IR 分层

> **决定**：走 Triton 的路子——`TTIR → TTGIR → LLVM IR → PTX → cubin`。
> **不做** source-to-source 的 CUDA C 后端（那是 TileLang / 转译器的路线）。

---

## 0. 为什么必须是这条链

出 CUDA C 看起来省事，但它有两个致命问题：

1. **它跳过了 TTGIR**——而 layout、warp 分布、`convert_layout` 全在 TTGIR 里。本项目要讲的 tiling 与 bank conflict，正好就是这一层的决策。跳过它，等于把最有价值的部分外包给 nvcc。
2. **它看不见真东西**：地址空间（global=1 / shared=3）、`barrier`、NVVM intrinsic、PTX、SASS——这些是"AI 编译器"面试真正会追问的地方，CUDA C 后端一律遮住。

代价是难得多。所以策略是：**复刻流水线的形状，缩小每一层的内容**（见 §6）。

## 1. 五层 IR

| 层 | 粒度 | 有 | 没有 | 对应 Triton |
|---|---|---|---|---|
| **TTIR** | 张量 | shape / dtype / 算术 / load-store | **无 layout、无线程概念** | TritonIR |
| **TTGIR** | tile + 线程 | **layout encoding**、warp 分布、smem 分配 | 无寄存器分配 | TritonGPU IR |
| **LLVM IR** | 每线程标量 | 地址空间、循环、`barrier`、NVVM intrinsic | 无 tile 概念 | LLVM dialect + NVVM |
| **PTX** | 指令 | 寄存器、`ld.shared`、`bar.sync` | — | PTX |
| **cubin** | 机器码 | — | — | — |

产物全部文本化落盘：`.ttir` / `.ttgir` / `.ll` / `.ptx`，**每一份都进 golden 测试**。

## 2. 十一个组件

| # | 组件 | 职责 | 允许决策？ |
|---|---|---|---|
| 1 | Frontend | AST → TTIR，只做语义落地 | 否 |
| 2 | TTIR 定义 + 形状/类型推导 | 广播、mask、归约后的 shape/dtype | 否 |
| 3 | Pass 框架 | 注册、排序、每 pass 前后 dump | — |
| 4 | **Tiling pass** | TTIR → TTGIR 的 block 形状（`BLOCK_*`、`num_warps`） | **是**（L1 tiling） |
| 5 | **Layout 代数** ★ | encoding 定义、`convert_layout` 生成、swizzle 选择 | **是**（bank conflict） |
| 6 | TTGIR 定义 | Value 上挂 layout、warp 映射、smem 表 | 否 |
| 7 | TTGIR → LLVM lowering | 标量化、地址空间、`barrier`、intrinsic | 否（机械） |
| 8 | LLVM 文本后端 + `llc` | 发 `.ll`，调 `llc -mtriple=nvptx64-nvidia-cuda` | 否 |
| 9 | cubin 驱动 | `ptxas` → cubin，`libcuda` 加载/launch | 否 |
| 10 | 对拍 / 差分 | 我 / 官方 Triton / torch 三方对照 | — |
| 11 | IR 文本化 | 序列化 + 读回，支撑 golden | — |

**只有 4 和 5 做决策**，其余全是机械变换。这条界线一旦模糊，性能变化就无法归因。

## 3. 六条铁律

1. **TTIR 里不许出现 layout**。出现即设计错——layout 是 TTGIR 的概念。
2. **跨 layout 必须显式 `convert_layout`**，禁止隐式转换。它是 smem 往返和 bank conflict 的唯一合法来源，必须能被数出来。
3. **地址空间由 TTGIR 决定**（global=1 / shared=3），LLVM 层不得自行猜测。
4. **codegen 不做决策**（同 §2）。
5. **每一层都能文本进、文本出**。
6. **block 形状只有 tiling pass 说了算**。

## 4. 五个现在定死、以后改不动的决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 是否上 MLIR | **不上**，自己写 IR，但**照抄分层**（TTIR/TTGIR 边界） | 4 周学 MLIR 写不出东西；但分层是 Triton 的精髓，必须复刻 |
| LLVM IR 怎么发 | **文本 `.ll`**，不用 llvmlite | llvmlite 不含 NVPTX target，发不出 GPU 代码 |
| 初期用几个 NVVM intrinsic | **1 个**：`llvm.nvvm.barrier0` | mma / cp.async 留 W3，先把链路打通 |
| TTGIR 先支持几种 layout | **1 种**：blocked（global）+ 固定 swizzle 的 shared | 先保证正确，W3 再让模型枚举 |
| `tl.dot` 怎么落 | W2 先**标量循环**（无 tensor core），W3 再试 `wmma/wgmma` intrinsic | 先把 tiling 讲清楚；mma 是 stretch |

## 5. 反直觉的好处：这条路在 Mac 上能验证得更多

`llc`（.ll → .ptx）和 `ptxas`（.ptx → .cubin）**都是纯 CPU 工具，不需要 GPU**。

所以 Mac 容器里能跑通：`IR 文本 → .ll → .ptx → .cubin →（可选）SASS 反汇编`。
**只有"执行 kernel"必须上 GPU。** 这比 CUDA C 路线的前置验证点更靠后，调试循环更短。

**S1 已实测的工具链事实**（2026-11-18）：

| 环境 | llc | ptxas | 结论 |
|---|---|---|---|
| Mac 本机 | Homebrew **LLVM 20.1.8**，含 `nvptx`/`nvptx64` | ❌ 无（macOS 无 CUDA toolkit） | 可本地出 `.ptx`，出不了 `.cubin` |
| 容器 `tritonlab-cpu` | Debian **LLVM 19.1.7**，含 NVPTX | ✅ triton wheel 自带 | 整条链可跑，`sm_90`/`sm_90a` 均支持 |

所以 `make smoke-cubin` 的策略是：本机 `llc`+`ptxas` 齐全则直接跑，否则借容器。

另一个实测教训：LLVM 函数必须用 **`ptx_kernel` 调用约定**（CC 71），否则生成的是 `.func` 而不是 `.entry`，**根本无法 launch**。

## 6. 里程碑（复刻形状、缩小内容）

| 周 | 目标 | Gate |
|---|---|---|
| **W0** | vector-add 打通五层：TTIR → TTGIR（结构平凡）→ `.ll` → `.ptx` → `.cubin` → 执行 | 数值对拍通过；`.ptx`/`.cubin` 能在 Mac 容器里产出 |
| **W1** | 行归约（`tl.max/sum/exp`）。**这一步会逼出真正的 TTGIR**（warp shuffle 还是 smem+barrier） | 对拍通过；TTGIR 里出现第一个真实的 layout 决策 |
| **W2** | GEMM：tiling + shared layout + 标量 dot（无 tensor core） | 数值正确（性能不比） |
| **W3** | Layout 代数 + 用 bank 模型枚举 swizzle + `convert_layout` 计数；ncu 对照 | 冲突预测 ≈ 实测；能解释每次 smem 往返 |
| **W4** | 第二个 workload + 三方差分对照 | 对照表齐全 |

## 7. 砍掉的东西（明确不做）

CUDA C 后端、TileLang 式 source-to-source、MLIR 依赖、autotune、fusion pass、动态 shape、多 kernel 流水、PTX 之后的 SASS 手工优化。

## 8. 风险

| 风险 | 对策 |
|---|---|
| `llc` 无 NVPTX target（Debian 包通常带，需验证） | W0 先验证；不可用则改用 `clang -target nvptx64` 或退到自研 PTX emitter（工期 +1 周，需重排里程碑） |
| NVVM intrinsic 名称随 LLVM 版本变动 | 锁定容器内 LLVM 版本，写进 ADR |
| `wgmma`（Hopper）需要较新 LLVM 支持 | 视为 stretch，W3 只做 `mma.sync`/`wmma`，不做 wgmma |
| 标量 dot 太慢，无法与官方对比性能 | 明确：W2/W3 比的是**可解释性**与**冲突计数**，不是 TFLOPS |
