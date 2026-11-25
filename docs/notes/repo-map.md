# 仓库地图：一个动作，三个世界

> 读法：每张图只加一层信息。看完 8 张图，你就知道每个文件为什么存在。

---

## 图 0：这个仓库是什么

**不是** kernel 库，**不是** Triton 的 fork。它是：

> 把 Triton 的"编译流水线"自己复刻一遍，用复刻过程理解 Triton。

所以它有三种东西混在一起：**代码**（复刻品）、**模型**（理解工具）、**资料**（对照物）。

---

## 图 1：三个世界

```
┌────────────── 编译世界（纯 CPU，Mac 就能跑）──────────────┐
│  .py ─▶ TTIR ─▶ TTGIR ─▶ LLVM dialect ─▶ .ll ─▶ .ptx ─▶ .cubin │
└──────────────────────────────────────────────────────────┘
                              │ 交接物：.cubin
                              ▼
┌────────────── 执行世界（必须有 GPU）─────────────────────┐
│  .cubin ─▶ libcuda 加载 ─▶ launch ─▶ 结果 ─▶ 对拍 torch      │
└──────────────────────────────────────────────────────────┘

┌────────────── 分析世界（纯 Python，无 IO）───────────────┐
│  smem 预算 / bank 冲突 / roofline / split-KV 启发式        │
│  作用：给编译决策提供"代价模型"，也给结果提供"理论下界"     │
└──────────────────────────────────────────────────────────┘
```

**关键设计**：编译世界与执行世界只通过一个 `.cubin` 通信。所以编译的 bug 只在 CPU 侧复现，执行的 bug 只跟驱动/参数有关——两类问题不会混在一起。

---

## 图 2：目录 = 三个世界 + 资料

```
proj-a/
├── minitriton/          编译世界：我们自己的编译器
│   ├── runtime/         ├─ 执行世界的桥（launch.py + toolchain.py）
│   ├── ttir/            └─ ⚠️ 过时：上 MLIR 之前的 dataclass IR 骨架
│   └── frontend/        ⚠️ 空目录（占位）
├── mlir/                编译世界：MLIR 方言与 pass（C++）
│   ├── include/minittir/ .td（唯一事实源）+ 手写头
│   ├── lib/             方言实现 + 插件入口
│   └── tests/           spike.mlir
├── analysis/            分析世界：四个纯 Python 模型
├── baseline/            手写基线（对照"编译器生成 vs 人手写"）
├── bench/               GPU 侧测量（带宽标定）
├── scripts/             GPU 侧脚本（端到端验证）
├── tests/               全部回归（零依赖 runner）
├── docs/                资料：spec / adr / notes
└── refs/                对照物：paged_attention.py + 整个 Triton 源码树
```

`refs/triton/`（1884 个文件）是**上游源码的只读副本**，用途是查证与对照，我们不编译它。

---

## 图 3：数据流走一遍（顺着数据走，别顺着目录走）

```
① Triton 子集 .py
      │  [minitriton/frontend/]  ← ⬜ 待建（S2 的任务，Python 只产 TTIR 文本）
      ▼
② TTIR 文本（.mlir）
      │  [mlir/] mlir-opt --load-dialect-plugin --load-pass-plugin
      │     • ttir-to-ttgir    ← ⬜ 未写（R2）
      │     • ttgir-to-llvm    ← ⬜ 未写（R3）
      ▼
③ LLVM dialect（.mlir）
      │  mlir-translate --mlir-to-llvmir
      ▼
④ .ll（LLVM IR 文本）
      │  [minitriton/runtime/toolchain.py] llc -mcpu=sm_90
      ▼
⑤ .ptx
      │  ptxas -arch=sm_90
      ▼
⑥ .cubin      ← ✅ S1 已打通（当前唯一用的是手写 hello.ll）
      │  [minitriton/runtime/launch.py] libcuda
      ▼
⑦ GPU 执行 → 结果
      │  [scripts/gpu_hello.py] 对拍
      ▼
⑧ PASS / results.json
```

**现在只有 ④→⑧ 是真的；①→③ 是待建的。**

---

## 图 4：每个文件一句话

### 分析世界（`analysis/`）

| 文件 | 一句话 |
|---|---|
| `hardware.py` | 设备档案：H100(sm_90/114SM/228KB) 与 A100(sm_80/108SM/164KB)，含 features（wgmma/tma 有无） |
| `smem_occupancy.py` | `smem = (BM·BK + BK·BN)·dtype·stages` → 每 SM 能驻几个 CTA → occupancy 上界 |
| `bank_model.py` | 给定布局算 bank 冲突；含 fp16 的 padding 单位陷阱 |
| `roofline.py` | 带宽下界、KV/权重交叉点（30/15）、split-KV 并行度启发式 |

### 编译世界（`minitriton/` + `mlir/`）

| 文件 | 一句话 |
|---|---|
| `minitriton/runtime/toolchain.py` | `.ll→.ptx→.cubin`，arch 显式参数，按架构分目录 |
| `minitriton/runtime/launch.py` | ctypes + libcuda：句柄管理 + ABI 适配 + 错误翻译 |
| `mlir/include/minittir/TtirOps.td` | 唯一事实源：`ttir` 方言 + 1 个 op `ttir.add` |
| `mlir/include/minittir/{TtirDialect,TtirOps}.h` | 手写胶水：把 lmgen 的 `.inc` 按正确顺序 include |
| `mlir/lib/TtirDialect.cpp` | 方言构造 + `addOperations<AddOp>` |
| `mlir/lib/TtirOps.cpp` | 展开 op 的 C++ 定义（一个 TU 一次） |
| `mlir/lib/Plugin.cpp` | 导出 `mlirGetDialectPluginInfo`，让 stock mlir-opt 认识我们 |
| `mlir/Makefile` | 4 次 tblgen + clang++ + `-undefined dynamic_lookup` |
| `mlir/tests/spike.mlir` | 输入：`%0 = ttir.add %a, %b : tensor<16xf32>` |

### 执行世界（`scripts/` + `bench/` + `tests/fixtures/`）

| 文件 | 一句话 |
|---|---|
| `tests/fixtures/hello.ll` | 22 行手写 IR：`out[tid]=(float)tid`，用 `ptx_kernel` CC |
| `scripts/gpu_hello.py` | S1 端到端：自动识别设备→选 arch→build→launch→逐元素对拍 |
| `scripts/gpu_verify.sh` | GPU 侧一键：带宽 + 全部测试 + JSON |
| `bench/bw_probe.py` | 实测 HBM 带宽 + 设备指纹（W0 的产出） |

### 基线（`baseline/`）

| 文件 | 一句话 |
|---|---|
| `baseline/gemm/v0_naive.py` | 用 Triton 写的最朴素 matmul，作为"人手写"的对照基线 |

### 资料（`docs/`）

| 文件 | 一句话 |
|---|---|
| `spec/architecture.md` | 五层 IR、十一个组件、六条铁律、五个定死的决策 |
| `spec/language-subset.md` | 语言子集契约：支持什么、不支持什么、报错要求 |
| `adr/0001-scope.md` | 范围与"不做什么" |
| `adr/0002-hw-baseline.md` | 硬件基线（H100 实测）+ 多卡策略 |
| `notes/mlir-setup.md` | out-of-tree MLIR 配方 + 四个坑 |
| `notes/triton-pipeline.md` | Triton 3.9.0 的六段流水线与 27 个 pass（含证据） |
| `notes/S1-walkthrough.md` | S1 逐段阅读指南（带行号） |
| `notes/S1-launch-illustrated.md` | launch.py 的图解版 |
| `notes/upstream-vs-mine.md` | 差异笔记模板（每个 pass 必填） |

### 回归（`tests/`）

| 文件 | 守住什么 |
|---|---|
| `run_local.py` | 零依赖 runner（无 pytest 也能跑） |
| `test_analysis.py` | 分析模型的数字（11 个） |
| `test_hardware.py` | H100/A100 档案差异（4 个） |
| `test_toolchain.py` | `.ll→.cubin` 可复现、arch 校验 |
| `test_launch.py` | launcher 可离线验证的部分 |
| `test_mlir_plugin.py` | stock mlir-opt 能加载我们的插件 |
| `test_ref_paged_attention.py` | paged KV 语义（6 个） |
| `test_gemm_v0.py` | 手写 Triton 基线（需要 triton，本地 skip） |

---

## 图 5：进度（按数据流标注）

```
① 前端           ░░░░░░░░░░   0%     ⬜ S2
② TTIR 方言      ██░░░░░░░░  20%     1 个 op（add）
③ ttir→ttgir     ░░░░░░░░░░   0%     ⬜ R2
④ ttgir→llvm     ░░░░░░░░░░   0%     ⬜ R3
⑤ .ll→.ptx→cubin ██████████ 100%     ✅ S1
⑥ libcuda 执行   ██████████ 100%     ✅ S1（H100 实测 68.4µs）
⑦ 对拍           ██████████ 100%     ✅
   MLIR 插件     ██████████ 100%     ✅ spike
   分析模型      ████████░░  80%     缺 A100 实测带宽
```

---

## 图 6：诚实状态（哪些是壳）

| 项 | 状态 |
|---|---|
| `mlir/` spike | **能用**：1 个 op，无 pass，无 layout，无 lowering |
| `minitriton/frontend/` | **空**：S2 才建（Python 侧只做解析，IR 全在 C++/MLIR） |
| `baseline/gemm/v0_naive.py` | 需要 triton；本地只 skip |
| `docs/oss/CONTRIBUTION-PLAN.md` | 已降级为非目标 |

---

## 图 7：四条阅读线（各约 30 分钟）

| 线 | 顺序 | 目的 |
|---|---|---|
| **A. 先建立整体感** | `README` → `docs/spec/architecture.md` → 本文件 | 知道在造什么 |
| **B. 走通 S1（已完成的真链路）** | `tests/fixtures/hello.ll` → `scripts/gpu_hello.py` → `minitriton/runtime/toolchain.py` → `launch.py:45-115` | 知道"能跑"是怎么做到的 |
| **C. 上车 MLIR** | `docs/notes/mlir-setup.md` → `mlir/include/minittir/TtirOps.td` → `mlir/lib/Plugin.cpp` → `make -C mlir spike` | 知道方言怎么被宿主认出来 |
| **D. 对照上游** | `docs/notes/triton-pipeline.md` → `refs/triton/include/triton/Dialect/Triton/IR/TritonOps.td` | 知道我们抄的是哪一层 |

---

## 图 8：一页纸心智模型

```
             ┌──────────── 理解 triton 的手段 ────────────┐
             │  自己造一个     +     对照上游源码           │
             └────────┬───────────────────────┬──────────┘
                      ▼                       ▼
   analysis/ （代价模型）        docs/notes/ （差异笔记）
        │                              ▲
        ▼                              │
   mlir/ 方言与 pass ──▶ .ll ──▶ toolchain ──▶ launch ──▶ 对拍
        （编译世界）              （执行世界）
```

**一句话**：`analysis/` 是"为什么这么选"，`mlir/` 是"把选择做出来"，`runtime/` 是"让它跑起来"，`docs/notes/` 是"我学到了什么"。
