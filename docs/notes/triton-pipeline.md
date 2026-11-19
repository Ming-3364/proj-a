# Triton 3.9.0 的真实流水线（含可核对证据）

> 证据来源：本机容器内 `pip show triton` 的 3.9.0 安装。
> Python 层直接读源码；C++ 层的 pass 名从 `_C/libtriton.so` 的字符串表抽取。

## 六段

```
① 前端              python/triton/compiler/code_generator.py  (ASTVisitor)
   .py ──▶ TTIR     ops: tt.load / tt.store / tt.dot / tt.reduce / tt.func
                    ↑ 这是"张量级"，没有 layout、没有线程

② TTIR passes      复用 MLIR 通用 pass：canonicalize / combine / inline / unroll

③ TTIR → TTGIR     convert-triton-to-tritongpu        ★ layout 在这里诞生
                   给每个 tensor 分配 #ttg.blocked；dot operand 变 #ttg.dot_op；
                   要进 smem 的变 #ttg.shared（带 vec/perPhase/maxPhase）

④ TTGIR passes     27 个（见下表），主干 6 个

⑤ TTGIR → LLVM     convert-triton-gpu-to-llvm         ★ 地址空间 / barrier / mma
                   产出 MLIR 的 LLVM dialect

⑥ LLVM → 机器码     mlir-translate --mlir-to-llvmir → .ll
                   llc -mtriple=nvptx64 -mcpu=sm_90   → .ptx
                   ptxas --gpu-name=sm_90             → cubin
```

后端入口在 `python/triton/backends/nvidia/compiler.py`：`make_ttir / make_ttgir / make_llir / make_ptx / make_cubin`，`binary_ext = "cubin"`。
`ptxas` 是通过 `subprocess.run` 调的——**和我们 `toolchain.py` 的做法完全一样**。

## TTGIR 的 27 个 pass（该版本真实存在）

主干（我们只需要理解这几个）：

| pass | 干什么 |
|---|---|
| `tritongpu-coalesce` | 为 global 访存挑 blocked layout，目标是 128B 对齐 |
| `tritongpu-remove-layout-conversions` | 删掉多余的 `convert_layout`（省钱的关键） |
| `tritongpu-accelerate-matmul` | 选 MMA 版本，把 dot operand 降成 `#ttg.dot_op` |
| `tritongpu-optimize-dot-operands` | 优化 smem→寄存器 的 operand 搬运 |
| `tritongpu-pipeline` | 软件流水：`num_stages` → 多缓冲 + `cp.async`/TMA |
| `tritongpu-reorder-instructions` | 指令重排（延迟隐藏） |

特化/边缘（**我们不需要抄**）：
`accelerate-matmul` 之外还有 `F32DotTC`、`assign-latencies`、`allocate-warp-groups`、
`automatic-warp-specialization`、`warp-spec-lowering`、`optimize-partition-warps`、
`schedule-loops`、`partition-loops`、`partition-scheduling`、`fuse-nested-loops`、
`hoist-tmem-alloc`、`promote-lhs-to-tmem`、`optimize-accumulator-init`、
`reduce-data-duplication`、`global-scratch-memory-allocation`、`coalesce-async-copy`、
`prefetch`、`combine-tensor-select-and-if`、`normalize-ws-while-loops` 等。

## 三条设计教训

1. **layout 不是"优化 pass"，而是 conversion 的产物。**
   TTIR 里没有 layout，到了 GPU 上不能没有 layout，所以 `convert-triton-to-tritongpu`
   必须当场给出布局。我们原计划把 layout 当独立 P2 pass，实际应理解为"转换即决策"。

2. **大量 pass 是在给前一个 pass 擦屁股。**
   `remove-layout-conversions`、`optimize-dot-operands` 都是在消除转换引入的 smem 往返。
   真实的编译器不是一条直线，而是"决策 → 修复 → 再优化"。

3. **最小可用集合只有 5~6 个 pass。**
   27 个里大部分是 Hopper/Blackwell 特化与微调。minitriton 抄 6 个就够讲清全部核心概念。

## 抄什么 / 不抄什么

| 抄 | 理由 |
|---|---|
| TTIR / TTGIR 两层 + 一次 conversion | 这是 Triton 的精髓 |
| `coalesce` / `accelerate-matmul` / `pipeline` / `remove-layout-conversions` | 覆盖"决策 + 擦屁股"两类 |
| `convert-triton-gpu-to-llvm` 的思路（地址空间 + barrier + intrinsic） | S2 的 lowering 目标 |
| `mlir-translate` + `llc` + `ptxas` 的 subprocess 调用 | 我们 S1 已经这么做了 |

| 不抄 | 理由 |
|---|---|
| 27 个 pass 全抄 | 大部分与 sm_90 无关 |
| warp specialization / tmem / 分布式 scratch | Blackwell 专属 + 复杂度高 |
| 多后端（AMD/Intel）抽象 | 与目标无关 |

## 一个值得一提的差异

Triton 的 launcher 不只是 ctypes：它有 `driver.c` + `make_launcher.py`，运行时生成并编译一个
小 C 启动桩，以减少每次 launch 的 Python 开销。我们的 `launch.py` 是纯 ctypes（简单、够用），
这个差异值得写进 `docs/notes/upstream-vs-mine.md`。
