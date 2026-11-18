# 路线图：minitriton —— 自研迷你 Triton 编译器

> **目标**：吃一段 Triton 子集源码，输出能在 GPU 上跑的 CUDA C++。
> **tiling 与 shared memory 布局由编译器的 pass 推导出来，不是我手填的。**
> 官方 Triton 只作对照基准（数值 / 性能 / 生成代码 / IR 结构 / ncu 指标）。

**判据（一句话）**：如果 `BLOCK_M` 和 swizzle 是我在源码里手写的，那是写 kernel；如果它们由 pass 从 IR 推出来，才是编译器。

---

## 0. 任务定义：一个动作，三层产出

| 层 | 动作 | 产出 | 设 Gate？ |
|---|---|---|---|
| ① **构建**（目标·一半） | 自己写 frontend / IR / passes / lowering | 能编译子集并在 GPU 上跑对 | **是**，唯一有 Gate 的 |
| ② **理解**（目标·另一半） | 每个 pass 做完，与上游 TTGIR / 实现对照，写一页差异笔记 | `docs/notes/upstream-vs-mine.md` | **是**，每个 pass 必交 |
| ③ ~~贡献~~（**已移出目标**） | 上游 issue / PR | — | 不设 Gate；顺手发现只记进笔记，不投专门时间 |

一句话：**构建产生问题，对照产生理解。**
贡献不再是交付物——它的位置被"差异笔记"顶替了：理解不落成文字，最后无法自证。

> 目标定义见 `TARGET.md`；上游贡献材料已降级到 `docs/oss/CONTRIBUTION-PLAN.md`（默认不启动）。

### 一个方法上的纠正：用「对照」，不要用「参考」

"参考上游实现"有个陷阱：照着写等于翻译，既不产生理解，也发现不了任何问题（因为你把它的 bug 一起抄了）。

正确顺序是 **先自己设计 → 再对照上游 → 差异才是收获**。差异分三类，各有归档方式：

| 差异类型 | 处理 |
|---|---|
| 上游更好 | 学到它的取舍，写进笔记（这是"理解"的正文） |
| 我更好 / 上游有疑点 | 候选贡献 → 进差分测试 → issue |
| 只是不同 | 记录 trade-off，不评判优劣 |

所以 W2~W3 每个 pass 收尾时要回答同一个问题：**"上游在这里怎么做的？为什么和我不一样？"** 答不上来，说明这个 pass 还没做完。

---

## 1. 对照标准：拿官方 Triton 比什么

这是本项目的"评分表"。没有这张表，"我自己写了个编译器"就没有说服力。

| 维度 | 比法 | 产出 |
|---|---|---|
| 数值 | 同一算法的官方 triton kernel 与生成代码，都对 torch reference 对拍 | allclose 通过表 |
| 性能 | 同 shapes / 同 tile 参数，latency 三方对照（我的 / 官方 triton / torch） | 折线图 |
| 生成代码质量 | 行数、smem 字节、寄存器数、SASS 指令构成（`LDS`/`LDSM`/`HMMA`） | 对照表 |
| **IR 结构** | 我的 Tile IR vs 官方 TTGIR：**同一个 kernel 两边的 pass 决策差异** | 差异表（面试杀手锏） |
| 硬件真实性 | ncu 的 `l1tex__data_bank_conflicts_*` vs 我的 bank 模型预测 | 误差表 |
| 编译器行为 | 官方为什么选这个 swizzle / 这个 stage 数，我的 pass 选了什么、为什么不同 | 分析小节 |

> 重点不是"比官方快"，而是**能解释差异**。比官方快 10% 但说不出原因，不如慢 20% 但每个差异都有归因。

---

## 2. 架构：复刻 Triton 的 IR 分层

```
Triton 子集源码 (.py)
  │  frontend（ast 解析 + constexpr 展开 + 形状/类型推导）
  ▼
TTIR ── 张量级，**没有 layout、没有线程**
  │  P1 tiling：选 BLOCK_*（受 smem 预算 + occupancy 约束）
  ▼
TTGIR ── tile + 线程：**layout encoding、warp 分布、smem 分配**
  │  P2 layout：blocked（global）/ shared 编码 + XOR swizzle（bank 模型驱动）+ convert_layout
  │  P3 mem plan：smem 生命周期、num_stages 多缓冲
  ▼
LLVM IR (.ll 文本) ── 每线程标量代码 + 地址空间(global=1/shared=3) + barrier + NVVM intrinsic
  │  llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90
  ▼
PTX ── ptxas -arch=sm_90 ──▶ cubin
  │
  ├─ 本地（容器内，**不需要 GPU**）：.ttir/.ttgir/.ll/.ptx/.cubin 全部能产出，可做 golden 与反汇编
  ▼
GPU：libcuda 加载 cubin → launch → 对拍 torch / 对比官方 Triton / ncu 取证
```

**四个关键决策**：

1. **不上 MLIR，但照抄它的分层**。自己写 IR（Python dataclass），但 TTIR / TTGIR 的边界必须复刻——那是 Triton 的精髓；4 周学 MLIR 只会让你写不出东西。
2. **后端落 LLVM IR，不落 CUDA C**。CUDA C 是 TileLang 式 source-to-source，会跳过 TTGIR 与地址空间/intrinsic/PTX 这些"AI 编译器"真正会被追问的层。
3. **LLVM 用文本 `.ll`**（不用 llvmlite：它不含 NVPTX target）。好处是 `.ll → .ptx → .cubin` 全在 CPU 上完成，Mac 容器里就能验证。
4. **MMA 分两步**：W2 先标量循环（无 tensor core，保证"对"），W3 再试 NVVM 的 `mma.sync`/`wmma` intrinsic；`wgmma`（Hopper）列为 stretch。

---

## 3. 支持的 DSL 子集（边界必须写死，否则 4 周必爆）

策略：**不支持就明确报错，绝不静默降级**。报错信息本身就是交付物（说明我知道边界在哪）。

| 支持（v1） | 说明 |
|---|---|
| `@triton.jit` kernel + `tl.program_id` / `grid` | 1D/2D 网格 |
| `tl.arange` / `tl.load` / `tl.store`（含 mask、`other=`） | 指针算术限定为"基址 + 线性索引表达式" |
| `tl.dot(a, b)` | 2D×2D，fp16/fp32 累加 |
| `tl.max` / `tl.sum`（axis 归约）、`tl.exp`、基本算术 | 覆盖 softmax |
| `for k in range(0, N, STEP)` | 编译期可展开或转成 device 循环 |
| `tl.constexpr` meta 参数 | 编译期常量，是 tiling 的输入 |
| `num_warps` / `num_stages` | 由 tiling pass 决定，也可手填覆盖 |

暂不支持（明确报错）：`tl.trans` 以外的复杂布局变换、atomic、`make_block_ptr`、TMA/descriptor、`inline_asm`、warp 特化、多返回值、数据相关的控制流。

---

## 4. 各阶段与验收（这是"编译器"含金量所在）

| 阶段 | 输入 → 输出 | 谁的决策 | 验收标准 |
|---|---|---|---|
| **P0 前端** | AST → TTIR | 形状/类型推导、meta 展开 | TTIR dump 与手写预期一致（golden） |
| **P1 tiling（L1 tiling）** | TTIR | 在 smem 预算与 occupancy 目标下选 `BLOCK_*`、`num_warps` | 模型预测的 occupancy 与 ncu 实测吻合（误差 <10%） |
| **P2 layout（bank conflict）** | TTIR → TTGIR | global 用 blocked；smem 选编码 + **XOR swizzle（用 `analysis/bank_model.py` 枚举）+ 显式 `convert_layout`** | ncu `bank_conflicts` ≈ 0，且与模型预测一致 |
| **P3 内存规划 + 流水** | TTGIR | smem 生命周期、`num_stages` 多缓冲 | PTX 里出现 `cp.async`，stall 下降 |
| **P4 TTGIR → LLVM** | TTGIR → `.ll` | 标量化、**地址空间**(global=1/shared=3)、`barrier0` | `.ll` 能被 `llc` 接受，PTX 语义正确 |
| **P5 llc + ptxas + driver** | `.ll` → `.ptx` → `.cubin` → 执行 | 无（纯工具链调用） | cubin 能在 H100 上跑出正确结果 |

**每个 pass 都必须能 dump 前后 IR**——这既是调试手段，也是面试时"我真的写过 pass"的证据。

---

## 5.0 增量切片：每步一个可独立运行的产物

**原则：纵向切，不横向铺。** 按层建（IR → IR → IR → 后端）会让前三周都没有一条能跑通的路径；
纵向切则第 1 天就有可验证的产物，并且能尽早锁死最贵的接口（TTGIR 的 layout 表示）。

| 步骤 | 产物（可运行的命令） | 验收 | 依赖 | 工时 |
|---|---|---|---|---|
| **S1 hello-cubin** | `make smoke-cubin`：手写 `.ll` → `.ptx` → `.cubin`；`scripts/gpu_run.sh hello` 在 H100 上跑出结果 | 本地能出 cubin；GPU 返回正确值 | 无 | 3~4h |
| **S2 TTIR→LLVM（vector-add）** | `python -m minitriton emit --fixture vector_add` 产出 `.ttir/.ttgir/.ll/.ptx/.cubin`；`gpu_run` 对拍 torch | 数值对拍通过 | S1 | 5~6h |
| **S3 前端** | `minitriton compile examples/vector_add.py --meta BLOCK=1024` 端到端 | 对拍通过（输入换成真实 Triton 源码） | S2 | 6~8h |
| **S4 归约 → 真实 TTGIR** | `softmax_row.py` 编译并跑对；TTGIR 里出现 warp shuffle 或 smem+barrier | 对拍通过 + 差异笔记首篇 | S3 | 6~8h |
| **S5 GEMM + tiling** | `matmul.py` 编译并跑对；TTGIR 有 block 形状与非平凡 layout | 数值正确（不比性能） | S4 | 8~10h |
| **S6 layout/swizzle** | swizzle 由 bank 模型枚举；`.ptx` 与 ncu 冲突计数对照 | 预测 ≈ 实测 + 差异笔记 P2 | S5 | 8~10h |

### 两个轴：宽度（能表达多少）与深度（决策多聪明）

```
深度 ↑
     │                        ┌─ S6 layout/swizzle（决策变聪明）
     │        ┌─ S5 GEMM ─────┤
     │  S4 归约┘              └─ P3 流水
     │
     │  S1 工具链 ─ S2 IR→LLVM ─ S3 前端（链路打通）
     └────────────────────────────────────────→ 宽度（支持的语句/算子）
```

**准则：宽度由 workload 拉动（pull），不要由"看到 `tl.xxx` 就补"推动（push）。**
理由：pass 优化需要一个真实 workload 才能被检验。没有 GEMM 就做 layout 决策，等于在真空里调参。
所以 S4/S5 是"为了让两个目标 workload 能写出来"而扩展宽度，S6 才回到深度。

**为什么是这个顺序**：

1. **S1 先做工具链，不做 IR**。PTX 版本、驱动兼容、launcher 的参数编排——这些是"环境级"失败，越早撞越好，而且它是后面每一步都要用的横切能力。它也是唯一能在**没有编译器**的情况下交付的产物。
2. **S2 故意跳过前端**。前端是"翻译"，价值低于 lowering；先用 fixture 构造 IR，证明"IR → 能跑的 cubin"，避免 parser 的形状绑架 IR 设计。
3. **S3 才接真实输入**。此时 lowering 已稳定，前端只需做语义落地，两者的 bug 不会混在一起。
4. **S4 是第一个"非平凡"步骤**：归约会逼出真正的 layout 决策（warp shuffle vs smem+barrier），从这一步起每个 pass 都要交差异笔记。

**每步的 Definition of Done**：`make test` 全绿 + 上述命令可复现 + （S4 起）一篇差异笔记。

**进度**：S1 的本地一半已完成 —— `make smoke-cubin` 产出 `build/hello.ptx`(838B) 与 `build/hello.cubin`(3128B)，
并纳入回归（容器内 26 passed）；剩余一半是 `libcuda` launcher 与首次上卡验证。
**每步回退的代价**：S1~S3 任一步失败，只影响该步；S4 之后的失败才可能需要重排里程碑。

---

## 5. 里程碑（4 周 × 10~12h）

| 周 | 做什么 | 交付 / Gate |
|---|---|---|
| **W0** | 子集 spec、TTIR/TTGIR 定义、CLI、以及**五层打通**：TTIR → TTGIR（结构平凡）→ `.ll` → `.ptx` → `.cubin` → 执行 | **Gate：vector-add 端到端跑对**，且 `.ptx`/`.cubin` 在 Mac 容器里能产出 |
| **W1** | 前端 + 行归约（`tl.max/sum/exp`）——**这一步会逼出真正的 TTGIR**（warp shuffle vs smem+barrier） | Gate：对拍通过；TTGIR 出现第一个真实 layout 决策 |
| **W2** | GEMM：P1 tiling + shared layout + 标量 dot（无 tensor core） | **Gate：数值正确**（性能不比）→ 最小可交付点 |
| **W3** | Layout 代数 + bank 模型枚举 swizzle + `convert_layout` 计数；mma intrinsic（stretch） | Gate：冲突预测 ≈ ncu 实测；每次 smem 往返可解释 |
| **W4** | 第二个 workload（attention 片段）+ 全面对照 + 报告 | Gate：§1 的六项对照表齐全 |

**砍法（必须提前定好）**：W3 结束若 Gate 未过 → W4 砍掉第二个 workload，只交 GEMM + 完整对照，并在报告里写明"generality 未验证"。

**诚实的工期判断**：4 周能到"两个 workload + 两个真 pass + 能对照出数"。要做到能编译任意 Triton 程序，是 6~8 周的事——所以子集边界就是纪律。

---

## 6. 新增目录（其余复用已有代码）

```
minitriton/
  frontend/     ast_parse.py  infer.py           # Triton 子集 → TTIR
  ttir/         core.py  ops.py  printer.py      # 张量级 IR（无 layout）
  ttgir/        layout.py  core.py  printer.py   # tile 级 IR（layout/线程/smem）
  passes/       tiling.py  layout_assign.py  memplan.py   # 三个真决策
  llvmgen/      lower.py  emit_ll.py             # TTGIR → .ll（标量化+地址空间+barrier）
  runtime/      toolchain.py  launch.py          # llc/ptxas 调用 + libcuda 加载
tests/          test_frontend.py  test_passes.py # golden：.ttir/.ttgir/.ll/.ptx
bench/          compare_official.py              # 三方对照（我/官方/torch）
```

---

## 7. 风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| 前端子集边界失控（想支持越来越多语法） | **高** | 用 §3 的表当契约；超范围直接 `UnsupportedError`，不实现 |
| `wmma` 的 smem 布局约束吃掉 W3 | 中 | fallback：保持标量 dot，只做 swizzle 与 ncu 对照（仍能讲故事 2） |
| 索引算术写错（layout → 线程映射）导致结果错 | 高 | 每个 pass 都有 IR dump；先在 elementwise 上验映射，再上 dot |
| 反复调试编译器太慢 | 中 | 本地只跑 golden 快照（毫秒级），GPU 只在 Gate 上跑 |
| 想和官方比"更快" | 中 | 明确不比速度，比**可解释性**（§1 末段） |
| 容器里的 `llc` 缺 NVPTX 后端 | 低 | W0 第一件事验证；镜像构建里已做成断言（`llc --version \| grep nvptx`），缺了直接构建失败 |
| NVVM intrinsic 名称随 LLVM 版本漂移 | 中 | 锁死容器内 LLVM 版本并写进 ADR；只用到 `barrier0` 一个 intrinsic 起步 |

---

## 8. 面试叙事（三个故事）

1. **"Triton 的 tiling 和 layout 我看懂了，因为我自己实现了一遍"** —— 能指着自己的 IR 说：block 形状是在 smem 预算与 occupancy 之间权衡出来的。
2. **"我的编译器有代价模型，不是硬编码 heuristic"** —— layout pass 用 bank 冲突模型枚举 swizzle，ncu 实测证实预测。
3. **"和官方对照的方法论"** —— 不是"我更快"，而是"同一 kernel 两边 IR 的 pass 决策差异在哪、为什么、谁更好"（这条全靠差异笔记支撑，不靠 PR）。

---

## 9. 复用清单（上一版 kernel-lab 计划留下的资产，角色已变）

| 已有 | 新角色 |
|---|---|
| `analysis/bank_model.py` | **P2 layout pass 的判据**（编译器内部决策依据） |
| `analysis/smem_occupancy.py` | **P1 tiling pass 的代价模型** |
| `analysis/roofline.py` | 判断生成 kernel 离硬件下界多远 |
| `refs/paged_attention.py` | 生成代码的正确性裁判 |
| `kernels/gemm/v0_naive.py` | 手写基线（对照"编译器生成 vs 人手写"） |
| `bench/bw_probe.py` + `docs/adr/0002` | 硬件标定（roofline 分母） |
| `tests/run_local.py` | 零依赖 runner，继续用 |
| ~~k0→k4 attention 阶梯~~ | **丢弃**：那是 kernel 工程师的活，不是编译器 |

> 旧计划已归档到 `docs/archive/plan-kernel-lab-v1.md`。
