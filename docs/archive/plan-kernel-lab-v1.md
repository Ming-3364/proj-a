# 计划：Triton 算子实验台 + SGLang 轻量集成（面试项目｜算子 & AI 编译器方向，4 周）

> 一句话定位：**用 Triton 写一个 paged-KV 的 decode attention kernel，把它接进 SGLang 当 attention backend，并用"L1 tiling / bank conflict / TTGIR 布局"这条线解释清楚每一个性能决策。**
>
> 目标不是"跑通一个 kernel"，而是能在面试里回答：*为什么这个 tile、这个 layout、这个 stage 数，硬件上发生了什么，我怎么测出来的。*

---

## 0. 决策摘要（先看这个）

| 项 | 结论 |
|---|---|
| 主线 workload | **Decode attention**（paged KV + GQA + split-KV），head_dim=128，fp16 |
| 训练场 workload | **GEMM tiling ladder**（v0 naive → v1 smem tiling → v2 swizzle/grouped ordering），作为可控、可预测的"物理实验台" |
| 编译器理解载体 | TTIR→TTGIR→LLVM→PTX→SASS 的 dump + 解析；`#ttg.swizzled_shared` 编码 → bank 映射手算 → ncu 实测对照 |
| 系统落地 | `register_attention_backend("mytriton")` 接进 SGLang 0.5.x；**定位是"轻量落地证明"，不是 backend 调优**（见 §3-D） |
| 目标岗位 | **算子 & AI 编译器**（已确认）→ W3 主攻 TTGIR/生成代码/cost model，W4 只留 3~4h 给 SGLang 落地 |
| GPU 资源 | **H100 PCIe 80GB（sm_90a）**，ncu 可用，集群可直连 PyPI（已确认）→ **TMA 路线锁定**（§11-B）；注意 PCIe 版带宽/ SM 数均低于 SXM，§5.2 已按 PCIe 重算 |
| 本地（Mac） | 只做：kernel 源码、参考实现、**Triton 解释器模式跑正确性（Linux 容器内）**、无 GPU 的分析模型、CI、报告生成 |
| 远端（GPU） | 只做：真实性能、ncu 指标、CUDA baseline 编译、SGLang 落地验证 —— 一条命令产出 JSON，回传到 Mac 出图 |
| 周期 | **4 周 × 10~12h/周（已确认，完整版）**；第 2 周末即形成可交付的最小闭环（切一刀就能拿去面试） |
| GPU 机时 | 公司集群/自有卡 → 走内部资源；若临时租卡，4090/A100 约 ¥1.5~6/h，全周期 15~25 卡时 ≈ ¥50~150 |

---

## 1. 事实核查（已在你这台机器上验证，不是猜的）

| 事实 | 影响 |
|---|---|
| `Python 3.13.13`（miniconda）、`torch 2.12.1` 已装，MPS 可用 | 参考实现、数值对比、图表都能在本地跑 |
| **macOS 装不上 triton**：PyPI 上 triton 3.9.0 只有 `manylinux_2_27/2_28` 的 x86_64 与 aarch64 轮子；清华镜像 `pip download triton` 返回 `from versions: none` | 本地**不能** `import triton`。所有"跑 Triton 代码"的动作必须在 **linux/arm64 容器**或远端 Linux 上做 |
| `docker` CLI 存在，但 **daemon 没起** | W0 第一件事：`colima start` 或开 Docker Desktop，否则本地连解释器模式都没有 |
| 无 `nvcc / ptxas / ncu / nsys` | CUDA baseline 只能"写 + 远端编译"，本地最多做语法自查 |
| `~/.ssh/config` 只有 `relay.baidu-int.com`、`icode.baidu.com` | 大概率是公司 CPU 开发机：**可能可以当"Linux CPU 跑解释器模式"的备选**，但不是 GPU |
| 磁盘剩余 37Gi | 够（GPU 侧不在这台机器上占盘） |
| 版本坐标：triton `3.9.0`、torch `2.15.0`、sglang `0.5.23`（当前最新） | 你本地 torch 2.12.1 落后于最新；GPU 侧统一 pin 一套，见 §7 |

### 1.1 已确认的环境（第二轮）

| 项 | 结论 | 对计划的影响 |
|---|---|---|
| GPU | **H100 PCIe 80GB（sm_90a）** | TMA 可行 → §11-B 锁定 TMA；但**带宽约为 SXM 的 60%**，所有 roofline 数字必须按新值重算（§5.2） |
| ncu | 可用 | §6 取证全部可做，bank conflict 能拿到真实计数（不必走 §10 的降级） |
| 集群出网 | pip 可直连 PyPI | 容器与 GPU 侧直接 `pip install triton==3.9.0`，无需内网镜像 |
| SGLang 版本 | pin PyPI `0.5.23`，**不追新特性** | 集成只作"落地探针"（§3-D），不依赖任何新算子/新后端 |

**一句话：这台 Mac 是"写代码 + 做数学 + 出报告"的机器，不是"跑 kernel"的机器。计划必须围绕这条硬边界来设计，否则第 3 天就会卡住。**

---

## 2. 为什么这个项目在面试里值钱

一个"我写了 Triton kernel"的项目满地都是。这个项目的差异化在于它同时证明三件事，而这三件事恰好是 AI 编译器/推理系统岗的三层能力：

1. **编译器层**：能读 TTGIR，理解 layout 是怎么被传播和选择的，知道 `tl.dot` 的 operand 为什么要过 shared memory、`convert_layout` 为什么会插进来。
2. **硬件层**：能用 roofline 判断一个 kernel 是算力瓶颈还是带宽瓶颈，知道 smem 预算怎么算、bank 怎么映射、occupancy 为什么上不去。
3. **系统层**：能把 kernel 塞进 SGLang 的真实调度里（paged KV、CUDA graph、continuous batching），并解释端到端指标为什么变化。

第 3 点是最稀缺的：绝大多数候选人止步于"kernel 比 torch 快 X 倍"。（本项目按你的岗位定位只做**最小验证**，见 §3-D——能接上、能自洽解释即可，不追求把 backend 调到极致。）

---

## 3. 技术主线：三个"实验台" + 一个落地

### A. 训练场：GEMM tiling ladder（第 1 周）

不是目的，是**标定工具**：GEMM 的算术强度、访存模式、tile 尺寸效应都可由第一性原理预测，用它来验证我的分析模型对不对，再去碰 attention。

| 版本 | 内容 | 要证明的事 |
|---|---|---|
| v0 | 每个 program 一个 `BLOCK_M×BLOCK_N` 输出块，直接 `tl.load` A/B 做 `tl.dot` | 基线：访存放大 ~BLOCK_N/BLOCK_K 倍，走 L1/L2 |
| v1 | 经典 smem tiling，`BLOCK_M=BLOCK_N=128, BLOCK_K=32/64`，`num_stages=2..4` | **L1 tiling 的正解**：全局访存降到 O(1/tile)，pipelining 掩盖延迟 |
| v2 | grid 的 **grouped ordering**（`GROUP_M=8`）+ 非方 tile（128×256） | L2 复用：证明"L2 tile"是 grid 调度问题，不是 kernel 内问题 |
| v3 | `tl.dot` 的 operand layout 变体（A 转置 / B 转置 / `tl.trans`） | 引出 `convert_layout`：**这是 bank conflict 与 smem 流量的真正来源** |

验收：v3 vs v0 的 TFLOPS 曲线能被我 §5.3 的模型在 ±20% 内预测。

### B. 主线：paged decode attention（第 1~3 周）

选它的理由：
- **它天然是带宽题**，不是算力题 → 逼你把 roofline 讲清楚（面试高频）。
- KV cache 是 **paged 布局**（block table 间接寻址），`tl.load` 的地址是"向量化的 gather"，正好是 Triton 的强项也是难点。
- GQA 让 K/V 的 head 维和 Q 不等宽，布局转换明显。
- 长上下文必须 **split-KV + merge**，这是"并行度 vs 合并开销"的经典权衡。
- SGLang 里已有成熟实现可对照（`sglang/kernels/ops/attention/decode_attention.py`），**有 ground truth 才能证明我不是自嗨**。

阶梯：
| 版本 | 内容 | 关键决策 |
|---|---|---|
| k0 | 单 program 扫全 KV（无 split） | 正确性基线，暴露并行度不足 |
| k1 | KV 按 `BLOCK_N` 分块 + online softmax（online max/sum 递推） | smem/寄存器压力、`tl.max/tl.sum` 的 reduction 布局 |
| k2 | **split-KV**（`num_kv_splits`）+ 两级 merge | 并行度、中间 buffer（partial max/sum/acc）的带宽成本 |
| k3 | GQA 复用（一个 program 吃多个 Q head，共享 K/V tile） | 减少 K/V 重复读 → 带宽账直接改善 |
| k4 | autotune（`BLOCK_N`, `num_warps`, `num_stages`, `num_kv_splits`） | 配置空间剪枝 + 用模型先验减少搜索 |

**每个版本都要出一行数： latency / 有效 HBM 带宽 / 相对理论下界的百分比。** 这张表就是面试的白板。

### C. 专项：bank conflict（贯穿，第 2 周集中）

**必须先讲清楚一个事实**（这是这个项目最容易被面试官"验货"的地方）：

> **Triton 不暴露 shared memory 布局。** 你不能像 CUDA 那样写 `__shared__ float s[32][33]` 来 padding 消冲突。smem 的分配、padding、XOR swizzle 全由后端在 TTGIR 的 `#ttg.swizzled_shared<{vec, perPhase, maxPhase, order}>` 里决定。

所以"在 Triton 里做 bank conflict 优化"是个伪命题，正确的做法是三件事：

1. **解析**：dump TTGIR，读出 shared encoding 的 `vec/perPhase/maxPhase/order`，用它手算每个 wavefront 落在哪些 bank 上，给出"冲突数 = f(tile 形状, dtype, swizzle 参数)"的闭式模型。
2. **对照实验**：构造会让冲突暴露的变体（转置访问、`tl.trans`、reduction 的跨 warp 归约、奇数宽度、`num_warps` 变化），用 ncu 计数验证模型。
3. **Ground truth**：写一个 ~60 行 CUDA kernel（`__shared__` padding 版 vs XOR swizzle 版），在 GPU 上编译，用 ncu 拿到真实的 `l1tex__data_bank_conflicts_*`，证明"硬件机制我懂，不只是会调 DSL"。

**为什么这个三段式在面试里强**：它把"我不会在 Triton 里控制 smem"从弱点变成了论点——我知道边界在哪，并且用可验证的方法绕过了它。

> 注意：在 Hopper/Blackwell 上，`tl.dot` operand 走的是编译器自动 swizzle 的路径，**大部分情况冲突已经是 0**。真正会看到冲突的是 **reduction（softmax 的 max/sum）和 layout conversion**。所以 §6 的实验矩阵把重心放在这两处，而不是 GEMM 的 dot operand —— 这是一个诚实的判断，也正好是 decode attention 的 k1 阶段。

### D. 落地：接进 SGLang（第 4 周，只做"轻量证明"）

已核对当前主线代码（`sglang@main`, v0.5.23 附近）：

- 注册机制：`sglang/srt/layers/attention/attention_registry.py` 里有
  ```python
  ATTENTION_BACKENDS = {}
  def register_attention_backend(name):
      def decorator(fn):
          ATTENTION_BACKENDS[name] = fn
          return fn
      return decorator
  ```
- `--attention-backend` 的合法值来自 `sglang/srt/arg_groups/choices.py` 的**静态列表** `ATTENTION_BACKEND_CHOICES`，并导出了 `add_attention_backend_choices = ATTENTION_BACKEND_CHOICES.extend`。
- 参考实现：`triton_backend.py`（约 2979 行，含 CUDA graph 变体、spec decoding、MLA），kernel 在 `sglang/kernels/ops/attention/decode_attention.py`。

**集成路径（W0 花 30 分钟验证，两条都留着）**：
- **A（推荐，稳）**：不碰 CLI。写 `my_serve.py`：先 `add_attention_backend_choices(["mytriton"])` + `register_attention_backend`，再用 `Engine`/`ServerArgs` 编程式启动。绕开 argparse 的静态 choices。
- **B（如 CLI 也想要）**：做一个 console script / `sitecustomize.py`，在 `sglang.launch_server` 之前 import 插件模块完成注册与 choices 扩展。

**实现范围严格限定为最小 backend**：读 `base_attn_backend.py` 拿到必须实现的方法（`init_forward_metadata` / `forward_decode` / `forward_extend` / CUDA graph 相关钩子），**只支持 decode + 单模型（Llama/Qwen 类 GQA）+ fp16 KV**，`forward_extend` 直接 fallback 到官方 triton backend。不做 MLA、不做 spec decoding、不做 fp8。**范围失控是这类项目最大的死因。**

> 因为你的目标是**算子 & AI 编译器**，这里只考核两点：① 我的 kernel 是否被真实调度链路正确调用（paged block table、`out_cache_loc`、seq_len 语义）；② 端到端 ITL 是否与独立 benchmark 的结论自洽。**CUDA graph 兼容性划为 stretch，不阻塞交付。**
>
> 版本策略：**直接 pin PyPI `sglang==0.5.23`**，不跟 main、不追新特性（DSA / MLA / fp4 那些与你无关）。用到的只有 `register_attention_backend` 和 `forward_decode` 两个点，越少依赖越好升级。

---

## 4. 双轨开发：Mac 与 GPU 的分工（本计划的核心设计）

### 4.1 Mac 上能做的（必须做到"零 GPU 依赖也能推进"）

| 事项 | 怎么做 |
|---|---|
| Triton 语法/语义正确性 | **Linux 容器 + `TRITON_INTERPRET=1`**：解释器把 kernel 逐 program 用 numpy/torch 跑，不需要 GPU。慢，但能抓 90% 的 bug（越界、mask、索引、dtype、online softmax 递推） |
| 数值参考 | torch 参考实现（`torch.einsum` + 显式 paged gather），fp32 vs fp16 容差表 |
| 分析模型 | 纯 Python：smem 预算、occupancy 估算、bank 映射与冲突计数、roofline 下界、split-KV 的最优切分数 |
| TTGIR 解析器 | 纯 Python 正则/AST 解析 `.ttgir` 文本（样例从 GPU 机或 CI 回传，也可先用仓库里存的 golden 文件） |
| CI / 自测 | `pytest` + marker `gpu`，本地 `-m "not gpu"` 全绿 |
| 图表与报告 | matplotlib + `RESULTS.md` 自动生成 |
| CUDA baseline | 只写代码，远端编译；本地做 clang-format 级别的静态检查 |

### 4.2 只能 GPU 做的（诚实边界）

真实延迟、TFLOPS、HBM 带宽达成率、ncu 的 bank conflict / wavefront / occupancy 计数、CUDA baseline 编译运行、SGLang 端到端 TTFT/ITL/throughput、CUDA graph 兼容性。

### 4.3 本地 Triton 环境（W0 必做，3 个备选）

```bash
# 方案 1（首选）：linux/arm64 容器，Apple Silicon 原生速度
docker build -f docker/Dockerfile.cpu -t tritonlab-cpu .
docker run --rm -it -v "$PWD":/w -w /w tritonlab-cpu \
  env TRITON_INTERPRET=1 python -m pytest tests/unit -q
```
`Dockerfile.cpu` 已经落地（`docker/Dockerfile.cpu`），实测踩到两个坑，都已写进文件注释：

| 坑 | 现象 | 解法 |
|---|---|---|
| Docker Hub 不可达 | `failed to resolve reference ... context deadline exceeded` | `BASE` 用镜像源（如 `docker.m.daocloud.io/library/python:3.12-slim`），`--build-arg` 可覆盖 |
| **torch 钉死 triton 版本** | `torch==2.9.1` 要 triton 3.5.x，与本项目 3.9.0 冲突 | `pip install --no-deps torch`，再手补 filelock/sympy/networkx/jinja2/fsspec |

另两个实测事实：
- PyPI 上 **aarch64 的 torch 只到 2.9.1**（x86_64 更新更快），别在 arm64 容器里 pin 最新版；
- triton 的 Linux wheel 里是 **AArch64 ELF** 的 `libtriton.so` 与 `ptxas`（我解开 wheel 用 ELF header 验过 `e_machine=0xB7`），
  所以 macOS 上即便硬装也 import 不了——这不是 pip 挑剔，是二进制格式不对。

- 方案 2：`~/.ssh/config` 里的公司 relay 机器（若是 Linux x86）→ 装 triton，`ssh` 里跑解释器模式。
- 方案 3：任意便宜 Linux CPU 云主机，`rsync` 后远程跑。

> 注意 `TRITON_INTERPRET=1` 的局限：**不建模内存层级、不建模并行、极慢**。它只保证"算得对"。所有性能结论必须来自 GPU。

### 4.4 产物回流（关键工程习惯）

```
Mac  ──git push──▶  GPU 机 ──make gpu-verify──▶ benchmarks/results/<gpu>/<kernel>_<sha>.json
 ▲                                                        │
 └──────────── rsync/git pull ◀───────────────────────────┘
```
- GPU 侧脚本**一条命令跑完**：正确性 → 基准 sweep → （可选）ncu → 写 JSON → 打印摘要。
- JSON 进仓库（体积小），Mac 上 `make report` 出图。**这样"在 Mac 开发、在 GPU 验证"不是口号，而是流水线。**

---

## 5. 技术要点（面试要能讲的部分）

### 5.1 "L1 tiling" 的分层定义与 Triton 映射

术语先对齐（面试官很可能就在这里追问）：

| 层级 | 含义 | 决定它的东西 | Triton 里的旋钮 |
|---|---|---|---|
| Register tile | 每个线程持有的 fragment | 后端根据 `#ttg.dot_op` 布局反向选择 | 不可直接控制（只能间接影响） |
| **L1 / SMEM tile** | 一个 CTA 复用的一块数据，放在 shared memory（L1 的程序员可管理部分） | `BLOCK_M/N/K` × dtype × `num_stages` | `BLOCK_*`, `num_stages` |
| L2 tile | 相邻 CTA 共享的工作集 | grid 的遍历顺序 | `GROUP_M` / pid swizzle |
| HBM | 只读一次的理想 | 上面三层是否做对 | — |

**smem 预算公式（要能白板推）**：
```
smem ≈ (BLOCK_M·BLOCK_K + BLOCK_K·BLOCK_N) · dtype_bytes · num_stages   (+ epilogue 缓冲)
例：128×128×32, fp16, 3 stages → (128·32 + 32·128)·2·3 = 49,152 B = 48 KiB
```
→ 立刻能回答"为什么 128×128×64 × 4 stage 会 OOM 或者掉 occupancy"，以及"为什么 128×256 的 tile 需要 opt-in 超大动态 smem"（Triton 会自动申请，但超过 48KB 后每 CTA 能并发的数量下降）。

**还要会说**：decode 阶段 batch 小、`BLOCK_M`（Q 的行数）很小 → GEMM 那套 tile 直觉**不适用**，主战场是 KV 那一维的分块 + 并行度。这就是"为什么需要 split-KV"。

### 5.2 Roofline：先算清楚这道题是带宽题

**先校准硬件参数。** 你的卡是 H100 **PCIe**，它和网上绝大多数 Triton 博客假设的 SXM 不是一回事：

| 参数 | H100 PCIe 80GB | H100 SXM（对照） | 影响 |
|---|---|---|---|
| HBM | 80 GB HBM2e | 80 GB HBM3 | 容量同、带宽不同 |
| 峰值带宽 | **≈2.0 TB/s** | ≈3.35 TB/s | 所有"下界"数字差 **1.7×** |
| SM 数 | **114** | 132 | 并行度上限、wave quantization |
| smem / SM | 228 KB | 228 KB | tile 预算结论可直接复用 |
| L2 | 50 MB | 50 MB | L2 tile 结论可复用 |

> **这条写进报告的"方法论"一节**：照抄 SXM 数字，会让你的"带宽达成率"凭空好 1.7 倍、并行度分析整体错位。**所以 W0 第一件事是 `bench/bw_probe.py` 实测带宽，所有公式一律代入 `BW_measured`，不代入数据表。**

模型（Llama-3-8B 类，fp16，32 层，8 KV head，head_dim=128）：

```
每 token 每层 KV = 2(K,V) · 8 · 128 · 2 B = 4,096 B
每 token 全部层  = 4,096 · 32        = 131,072 B = 128 KiB
4K 上下文单序列   = 128 KiB · 4096    = 512 MiB      (8K → 1 GiB)
权重             = 8.03e9 · 2 B      ≈ 16.1 GB
```
- **bs=1, 4K**：一步 decode 读 ~16.1 GB 权重 → 在 **2.0 TB/s 上 ≈8.0 ms 下界**（SXM 才是 4.8 ms）。KV 仅 0.5 GB。
- **交叉点与带宽无关**（两边同除 BW）：`b · KV_per_seq ≈ 权重` → 4K 时 **b≈30**（16.06e9 / 5.369e8），8K 时 **b≈15**。
  （注：别用 0.512 GB 去除——512 MiB = 5.369e8 B，我曾在这里算错过一次，`analysis/roofline.py` 现在由测试守住这个数。）
- **推论**：bs<16 的 decode 本质是"权重带宽题"。**我的实验必须落在 bs≥32 / ctx≥8K 的 KV-bound 区**——很多 kernel benchmark 就是在这里翻车的（测了半天，测的是权重读取）。

**PCIe 特有的坑：wave quantization。** 114 = 2×3×19，128/256/1024 这些常见 grid 都不是它的倍数。若每 SM 只能驻 1 个 CTA（大 smem tile 的常见情况），一个 128 CTA 的 grid 会被切成 114 + 14 两波，有效利用率仅 128/228 ≈ **56%**。这就是 E14 要测的，也是 persistent kernel 的直接动机。

（数字先进代码、进报告前一律替换为 `bw_probe` 与 `nvidia-smi` 的实测值。）

### 5.3 无 GPU 分析模型（Mac 上就能写，且是面试加分项）

纯 Python 三件套，输入 `(tile 形状, dtype, num_warps, num_stages, GPU 规格)`：

1. **smem/occupancy 模型**：smem 占用 → 每 SM 可驻留 CTA 数 → achieved occupancy 上界（对比 ncu 的 `sm__warps_active.avg.pct_of_peak_sustained_active`）。
2. **bank 冲突模型**：按 `#ttg.swizzled_shared` 的 `(vec, perPhase, maxPhase, order)` 把逻辑 `(row, col)` 映射到物理地址 → `bank = (addr/4) % 32` → 按 warp 内 32 lane 统计每个 wavefront 的 bank 冲突数。输出：给定 tile 的"冲突数预测"。
3. **Roofline + split-KV 模型**：给定 `(bs, ctx, heads)` 预算 HBM 时间下界 vs 计算下界，并求 `num_kv_splits` 的最优值（并行度收益 vs 部分结果读写成本）。

**模型的验收标准：在 GPU 实测的 20 组配置上，延迟预测误差中位数 < 25%。** 达不到就写"模型在哪些 regime 失效、为什么"——这比假装准确更有说服力。

### 5.4 度量口径（先把指标定死，避免自欺）

| 指标 | 来源 | 用途 |
|---|---|---|
| `latency_us` (p50/p90) | `triton.testing.do_bench` / CUDA events | 主指标 |
| 有效带宽 GB/s = bytes_moved / latency | 自己算 | 对比 HBM 峰值，判断离下界多远 |
| TFLOPS | 同上 | 只在 prefill/GEMM 有意义 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`<br>`..._op_st.sum` | ncu | **冲突的直接证据** |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` | ncu | 冲突的相对分母（wavefront 数） |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | ncu | occupancy |
| `dram__bytes_read.sum` | ncu | 验证"我以为的访存量"是不是真的 |
| smem/regs per thread | `PTXAS_OPTIONS="-v"` 的 ptxas 输出 | 解释 occupancy 天花板 |
| TTFT / ITL / output tok/s | SGLang bench | 端到端 |

---

### 5.5 编译流水线的"五个必看 pass"（算子/编译器岗的硬通货）

Triton 的降级链（`TRITON_KERNEL_DUMP=1` 拿最终各阶段产物；逐 pass 打印用 `MLIR_ENABLE_DUMP=1`，变量名以 3.9 实测为准）：

```
Python AST ──frontend──▶ TTIR ──▶ TTGIR ──▶ LLVM IR ──▶ PTX ──ptxas──▶ cubin/SASS
                              ↑ layout 与 smem 在这一层被决定
```

要能说出每个 pass 在干什么、什么情况下成为瓶颈：

| pass | 干什么 | 你会观察到它 |
|---|---|---|
| `tritongpu-coalesce` | 为全局 load/store 选定 `#blocked` 布局，目标对齐 128B 事务 | v0 matmul 里 A/B 的 load 布局 |
| `tritongpu-remove-layout-conversions` | 消除冗余 `convert_layout`（即避免多余的 smem 往返） | 加 `tl.trans` 后 IR 里"本该消失却没消失"的那些 |
| `tritongpu-accelerate-matmul` | `tt.dot` → `#ttgpu.dot_op` + MMA（`ldmatrix` / `wgmma` / `tcgen05`） | k1 的 `tl.dot` 处 operand encoding |
| `tritongpu-pipeline` | 软件流水：`num_stages` → 多缓冲 smem + `cp.async`/TMA | `num_stages` 一改就变的那段 |
| `tritongpu-optimize-dot-operands` | 优化 dot operand 从 smem 搬进寄存器 fragment 的路径 | E5/E9 里 dot operand 的 `convert_layout` |

**读 TTGIR 的三步法**（W3 前练成肌肉记忆）：
1. 先看 `tt.func` 的 `#blocked` / `#shared` 属性表 —— 这是"数据在哪一层、按什么形状排布"的唯一真相。
2. 再数 `ttg.convert_layout`：每个都可能是一次 smem 往返（除非被 remove-layout-conversions 消掉）。
3. 最后看 `tt.dot` 的 operand 编码与 MMA 形状，判断有没有命中 `ldmatrix`/`wgmma` 快路径。

**取证命令**（GPU 侧）：
```bash
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./dump TRITON_ALWAYS_COMPILE=1 python bench/run_gemm.py
PTXAS_OPTIONS="-v" TRITON_ALWAYS_COMPILE=1 python bench/run_decode.py   # smem / regs 报告
cuobjdump -sass dump/*.cubin | grep -E "LDS|LDSM|HMMA"                  # SASS 级取证
```

## 6. 实验矩阵（W2 集中做，一张表就是一个下午）

| # | 变量 | 对照 | 预期（写下来再验证） | 取证 |
|---|---|---|---|---|
| E1 | GEMM `BLOCK_K` 32→64→128 | 固定 128×128 | 32 太小（pipeline 填不满），128 smem 爆/掉 occupancy，64 最优点 | latency + smem + occ |
| E2 | GEMM `num_stages` 2→5 | 128×128×64 | 3~4 饱和，5 因 smem 反而退化 | latency 曲线 |
| E3 | GEMM pid 顺序：行优先 vs `GROUP_M=8` | 同 kernel | L2 命中率↑，大 M/N 时 1.2~2× | `lts__t_sectors` / latency |
| E4 | softmax reduction 的 `num_warps` 1/2/4/8 | BLOCK_N 固定 | 跨 warp 归约走 smem，warps 越多 wavefront 越多，冲突模式变化 | bank conflict 计数 |
| E5 | 同一 tile 的 `tl.trans` 前 vs 后 | — | 插入 `convert_layout`，smem 往返 + 冲突尖峰 | TTGIR diff + ncu |
| E6 | 奇数/非 2 幂宽度 tile（如 96、160） | 128 | swizzle 参数失配 → 冲突上升 | 模型 vs ncu |
| E7 | decode `num_kv_splits` 1/2/4/8/16 | ctx=4K/32K | 短 ctx 无收益（合并开销），长 ctx 显著；最优点由模型预测 | latency + 并行度 |
| E8 | GQA 复用（1 program 吃 4 个 Q head） | 不复用 | K/V 读取量 ÷4，带宽账直接改善 | `dram__bytes_read` |
| E9 | CUDA padding vs XOR swizzle vs Triton 自动 | 8×32 fp16 转置 tile | 三者冲突计数：18 / 0 / 0（预期） | ncu 三连 |
| E10 | `tl.multiple_of` / `tl.max_contiguous` 提示有无 | 合并访存 | 向量化宽度 4B→16B，latency 明显改善 | PTX 里 `ld.global.v4` |
| E11 | autotune 成本模型：20~30 组配置拟合 → 预测未见配置 | 全量网格搜索 | 剪枝后搜索次数降 ≥50%，预测误差中位数 <20% | 模型 vs 实测表 |
| E12 | SASS 取证：`cuobjdump -sass` 查 `LDS.128` / `LDSM` / `HMMA` | 向量化提示有无 | 提示后出现 128-bit smem 访问，且与 E10 的 PTX 变化对应 | SASS 文本 diff |
| E13 | **TMA**（`tl.make_tensor_descriptor`）vs 指针算术，同一 GEMM | 128×128×64 / 128×256×64 | TTGIR 里出现 `cp.async.bulk.tensor`，地址计算指令消失；内维 256 时需 128B swizzle 才不退化 | SASS + TTGIR + ncu |
| E14 | grid 128 vs 114 vs 114×2 的尾部效应；persistent 静态分配 | ctx=4K, bs=1/8/32 | 非整除时尾部损失可达 2×（1 CTA/SM 下）；persistent 抹平 | latency vs grid 扫描 |

**E9 是全项目的"钉子"**：它用一个 60 行 CUDA kernel 把"bank conflict 到底是什么"钉死，然后证明 Triton 的自动布局在多数情况下已经解决了它，但在 reduction/transpose 处仍会露出来。面试时讲这一段，可信度直接拉满。

---

## 7. 里程碑

### W0（3~5h）：把"能不能干"变成"确定能干"

- [ ] `colima start` 或 Docker Desktop，跑通 §4.3 容器 + `TRITON_INTERPRET=1` 的 hello-matmul
- [ ] `add_attention_backend_choices` / `register_attention_backend` 的最小验证（nginx 式：不启模型，只 import + 注册成功）
- [ ] **校准硬件**：`nvidia-smi` + `bench/bw_probe.py` 实测 HBM 带宽/SM 数/smem，数字写进 `docs/adr/0002-hw-baseline.md`（**别用数据表**，见 §5.2）
- [ ] 确认 `ncu` 对 toy kernel 采集成功，且 `torch.cuda.get_device_capability()` 报 `(9,0)`
- [ ] 仓库骨架 + `Makefile` + CI + 版本 pin
- [ ] 写 `docs/adr/0001-scope.md`：明确"不做什么"

**Gate：W0 结束若 GPU 仍无着落 → 立刻切 Plan B（见 §10），不要等。**

**W0 进度（2026-11-11，本地已完成）**：
- ✅ 仓库骨架：`Makefile`、`pyproject.toml`、`conftest.py`、`.gitignore`、`docker/Dockerfile.cpu`
- ✅ **参考实现（paged KV 语义）**：`refs/paged_attention.py`；6 个测试全绿；并做了**变异测试**——把 GQA 分组写成 tile、或不截断最后一个部分页，误差都是 ~2.0（容差 1e-5），证明测试不是空转
- ✅ 本地零依赖测试：`python3 tests/run_local.py`（macOS 无 pytest 也能跑；两个 kernel 测试在无 triton 时自动 skip）
- ✅ `bench/bw_probe.py`：待 GPU 侧执行 → 填 ADR 0002
- ✅ ADR：`docs/adr/0001-scope.md`（范围与"不做什么"）、`0002-hw-baseline.md`（待填模板）
- ⏳ **待 GPU**：`bw_probe` 实测带宽、`ncu` 采集验证、`register_attention_backend` 注册验证
- ✅ **容器链路已跑通（实测）**：`docker/Dockerfile.cpu` 构建成功，容器内 `python tests/run_local.py` → **23 passed / 0 skipped**
  （两个 GEMM kernel 测试在 `TRITON_INTERPRET=1` 下真实执行；把 grid 故意截断成 `(1,1)` 会报 1.2e+01 的误差，证明不是空转）
- ✅ 分析层（无 GPU 模型）落地：`analysis/{hardware,smem_occupancy,bank_model,roofline}.py`，11 个测试守住文档里的数字
  （顺带修正了文档里一个错误：KV/权重交叉点是 **30 / 15**，不是 31 / 16——`0.512 GB` 与 `512 MiB` 混用了）

### W1（10~12h）：GEMM ladder + 分析模型 v0 + 解释器测试

- [ ] v0/v1/v2/v3 四个 kernel + torch 参考 + 单测（解释器模式全绿）
- [ ] smem/occupancy/bank 模型 v0（先只做 GEMM）
- [ ] 首次 GPU run：E1/E2/E3 → 第一张 TFLOPS 曲线 + 模型误差

**交付：`RESULTS.md` 有第一张图，模型误差 < 40%（W3 再收紧到 25%）。**

### W2（10~12h）：decode attention k0→k2 + bank conflict 专项（E4~E6, E9）

- [ ] paged KV 参考实现（含 block table 语义）
- [ ] k0/k1/k2 + split-KV merge
- [ ] TTGIR 解析器 + swizzle→bank 映射
- [ ] CUDA baseline（padding / swizzle 两版）+ E9 三连 ncu
- [ ] roofline 一节写完（带宽账）

**交付（可拿去面试的最小闭环）：** 一个能讲 15 分钟的完整故事——"decode 是带宽题 → tile 决定 smem/occupancy → reduction 处有冲突 → 我用模型+实测证明了这一点 → split-KV 把并行度补上"。

### W3（10~12h）：编译器层吃透 + k3/k4 + cost model（**本项目重心**）

- [ ] k3 GQA 复用（E8）、k4 autotune；E7 / E10 / E12
- [ ] §5.5 的流水线逐 pass 走一遍，每个 pass 前后各留一份 IR → `docs/notes/ttgir-walkthrough.md`
- [ ] **TTGIR 精读 3 个案例**：① `tl.trans` 引入的 `convert_layout` 往返；② dot operand 为什么被选成 `dot_op` 而非 `shared`；③ reduction 的 `#shared` 往返与 bank 访问
- [ ] E11：autotune 成本模型（拟合 + 剪枝），剪枝率与误差都进结果表
- [ ] 模型误差收紧到 <25%，写"失效 regime"分析（哪类形状我的模型不管用、为什么）

**交付：`docs/notes/ttgir-walkthrough.md` —— 面试时可直接投屏的那一页。**

### W4（10~12h）：深挖一个硬件特性 + 轻量落地 + 收尾

- [ ] **TMA 深挖**（已锁定，sm_90a）：E13 的两种实现对照 + TTGIR 里 `cp.async.bulk.tensor` 是否出现、为什么没出现
- [ ] SGLang 最小 backend（decode only，eager 优先，extend fallback），只验证"被正确调用 + 端到端自洽"（限 3~4h）
- [ ] CUDA graph 兼容（**stretch**：够时间就做，不够就记为已知限制）
- [ ] 报告 + 5 分钟讲稿 + 10 个高频追问的答案

---

## 8. 目录结构

```
proj-a/
├── PLAN.md                      # 本文档
├── README.md                    # 一页纸：定位 / 怎么跑 / 结论三句话
├── RESULTS.md                   # 自动生成的结果表 + 图（面试直接发这个）
├── Makefile                     # dev / test / gpu-verify / report
├── pyproject.toml               # pin: triton==3.9.0, torch==2.12.x(cpu), numpy, pytest, matplotlib
├── docker/Dockerfile.cpu        # linux/arm64 + triton + torch-cpu（本地解释器模式）
├── kernels/
│   ├── gemm/{v0_naive,v1_tiled,v2_grouped,v3_transposed}.py
│   └── attention/{k0_naive,k1_tiled,k2_splitkv,k3_gqa,k4_autotune}.py
├── refs/                        # torch 参考实现（paged KV 语义在此定义）
├── analysis/
│   ├── smem_occupancy.py        # 预算 + occupancy 上界
│   ├── bank_model.py            # swizzle 编码 → bank 冲突预测
│   ├── roofline.py              # 带宽/算力下界 + split-KV 最优
│   └── ttgir_parse.py           # 解析 #ttg.swizzled_shared
├── cuda/baseline_{pad,swizzle}.cu   # bank conflict ground truth
├── sglang_plugin/
│   ├── backend.py               # register_attention_backend("mytriton")
│   └── serve.py                 # add_attention_backend_choices + 启动
├── bench/
│   ├── run_gemm.py  run_decode.py   # 统一 CLI，输出 JSON
│   └── ncu.py                        # 固定 metric 集合的中断式采集
├── benchmarks/results/<gpu>/*.json   # 回流的实测数据（进 git）
├── docs/{adr,notes,figures}/
└── scripts/{gpu_verify.sh,remote.sh,report.py}
```

---

## 9. 交付物 & 面试叙事

**交付物**：可直接跑的仓库 + `RESULTS.md`（4 张核心图：TFLOPS 曲线、decode 带宽达成率 vs 理论下界、bank conflict 模型 vs 实测散点、cost model 预测 vs 实测散点）+ `docs/notes/ttgir-walkthrough.md` + 5 分钟讲稿。

**四个故事（各 2 分钟；时间不够就只讲 2 和 3）**：
1. **"我用模型预测了硬件"**：无 GPU 时先建模，上卡后 20 组配置误差中位数 X%。→ 证明我不是调参碰运气。
2. **"Triton 不给我 smem 控制权，我换了个办法"**：解析 swizzle 编码 + CUDA ground truth + ncu 三连。→ 证明我懂边界且能绕。
3. **"autotune 是编译期问题，不是暴力搜索"**：成本模型 + 剪枝，搜索次数降 X%、误差 Y%。→ 直接对上"AI 编译器"岗的核心能力。
4. **"从 kernel 到系统（最小闭环）"**：接进 SGLang，ITL 与独立 benchmark 结论自洽；并坦白 CUDA graph 未做及原因。→ 证明有落地意识且不吹。

**会被追问、必须提前写好答案的**：
- 为什么 decode 用 split-KV 而 prefill 不用？
- `num_stages` 拉高为什么有时反而慢？（smem→CTA 数→occupancy）
- `tl.dot` 的 operand 为什么要经过 smem？直接寄存器不行吗？（mma 的 fragment 布局与 `ldmatrix`）
- 你怎么知道 ncu 的冲突计数对应我模型的哪一个 wavefront？
- 你的 backend 在 CUDA graph capture 下为什么没崩？指针稳定性怎么保证的？
- 官方 triton backend 比你快/慢的部分具体归因是什么？（**必须有归因，不能只说"接近"**）
- `tritongpu-pipeline` 在你的核心里插了几级？为什么不是越多越好？
- 你的成本模型用了什么特征？为什么它能泛化到没测过的配置？
- 如果允许你改 Triton 的一个 pass，你改哪个、为什么？

---

## 10. 风险与降级

| 风险 | 概率 | 触发信号 | 降级方案 |
|---|---|---|---|
| 拿不到能跑 ncu 的 GPU（部分云/MIG 禁 profiling） | 中 | W0 验证失败 | 只做 timing + `dram__bytes` 类基础指标；bank conflict 改为"模型 + CUDA 微基准"两者对照，明确标注"未取得 ncu 计数" |
| 本地 Docker 起不来 | 中 | `docker info` 失败 | 方案 2/3（公司 relay 机 / 便宜 CPU VM） |
| SGLang 版本漂移，backend 接口变动 | 中 | pin 的 commit 上接口对不上 | **pin git commit**，不跟 main；接口差异写进 ADR |
| 4 周做不完 | 中 | W2 末 k2 未通过 | 砍 W4 的 SGLang 落地与 §11-B 深挖，保留 kernel 层 + TTGIR 走读 + cost model（故事 1/2/3 全在） |
| 照抄 SXM 的带宽/SM 数字，导致所有结论偏移 1.7× | **高** | 报告里出现 "3.35 TB/s"、"132 SM" | W0 用 `bw_probe` 实测并把数字写进 ADR；代码里带宽一律走常量 `BW_MEASURED` |
| CUDA graph 集成翻车 | 中 | W4 前半 | 它已是 stretch：eager 跑通即可交付，graph 兼容性写进"已知限制" |
| 追求性能数字导致 scope 爆炸 | 高 | 出现"再优化一版" | 定死：**只交到 k4，性能目标 0.8~0.95× 官方 backend**，超出即停 |

---

## 11. 你问的"还有其他的吗"——建议追加的关键词

按"性价比 / 面试杀伤力"排序。标 ★ 的是我建议一定要塞进 4 周里的。

### A. 必加（直接强化主线，几乎零额外成本）
1. ★ **`convert_layout` / layout propagation**：TTGIR 里最值钱的概念。你的每一次 `tl.trans`、每次 `tl.dot` 前的不匹配，都由它买单。学会在 dump 里一眼看出"这里多了一次 smem 往返"。
2. ★ **software pipelining（`num_stages`）与 cp.async**：讲清楚"延迟隐藏"是怎么在 Triton 里自动发生的，以及它和 smem 预算的三角制约。
3. ★ **`tl.multiple_of` / `tl.max_contiguous`**：一行提示换来向量化宽度变化，PTX 从 `ld.global.b32` 变 `v4.b32`。**成本极低、演示效果极好**（E10）。
4. ★ **L2 tile = grid 调度**（`GROUP_M` / pid swizzle）：把"tiling"从 kernel 内扩展到 grid 层，是面试官区分"背过教程"和"想过"的分水岭。
5. **确定性归约（deterministic reduction）**：SGLang 有 `RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND` 这种概念。能讲"atomic add 的不确定性与批处理复现性"是系统岗加分。

### B. 选一个做深（"编译器如何适配硬件"的案例，4 周内只挑一个）
6. **TMA / `tl.make_tensor_descriptor`**（Hopper+）：硬件拷贝引擎替代 cp.async，smem 布局与 swizzle 的约束更强 —— 和 bank conflict 主题天然咬合。
7. **warp specialization**（Blackwell）：生产者/消费者 warp 分工，Triton 3.x 已支持。前沿、有话题性，但文档少、坑多。
8. **fp8 KV cache / 量化 attention**：SGLang 的真实需求；量化会改变 tile 的字节数 → 直接影响 smem 预算与冲突模式，能把你的模型用第二次。

> **已锁定：做 6（TMA）**，因为你的卡是 H100 PCIe / sm_90a —— 这是 TMA + wgmma 的主场，而且 TMA 的 **smem swizzle 变成显式约束**（32/64/128B），和 bank conflict 主题是一枚硬币的两面，面试时能连起来讲。
>
> **TMA 硬约束清单**（W4 的 checklist，别踩）：
> - 每个 box 维度 ≤ **256** 元素 → `BLOCK_M/BLOCK_N ≤ 256`，128×256 已接近上限；
> - 全局地址 **16B 对齐**，最内维字节数为 16 的倍数；
> - swizzle 只有 **32 / 64 / 128B** 三档，由内维字节宽度决定 —— 选错就退化；
> - device-side descriptor 需要 `triton.set_allocator(...)`（host / device 两套 API 别混用）；
> - 仅 sm_90+，且 **wgmma 需要 `sm_90a` 编译目标**，否则静默降级成 mma.sync。
>
> 8（fp8 KV）保留为 fallback：若 W3 结束 TMA 还没跑通，直接切它，不要再恋战。
> 7（warp specialization）本项目不做，但报告里写一段"换到 Blackwell 我会怎么做"——这题面试常问。

### C. 加分项（有余力再做，做了就是差异化）
9. **autotune 的成本模型**：用 20~30 组实测拟合一个 latency 预测器，展示"搜索空间剪枝把 autotune 时间降 X%"。这是"AI 编译器"味道最重的一小块，而且完全能在 Mac 上做。
10. **对比另一个 DSL**：SGLang 仓库里已经有 `helion/`、`cutedsl_*`、`tilelang` 生态的痕迹。花 2 小时用 TileLang 或 Helion 写同一个 decode kernel，写一页对比（表达能力/生成代码质量/调优手感）。**面试里几乎没人能聊这个。**
11. **PTX/SASS 级取证**：`cuobjdump -sass` 看 `LDS.128` / `LDSM` 有没有出现，验证"我以为的向量化"。
12. **Triton 编译缓存与 kernel 变体膨胀**：`TRITON_ALWAYS_COMPILE`、cache dir、SGLang 里 CUDA graph 变体数量 —— 工程视角的坑，很真实。

### D. 明确不要碰（时间黑洞，除非你投的就是编译器开发岗）
- 自己写 DSL 前端 / 改 Triton 的 pass / 自己实现 layout 推断。
- 从零写 flash-attention 的 backward。
- 多卡并行、PD 分离、投机解码集成。

---

## 12. 确认状态

**已全部确认（2026-11-11）**：H100 PCIe 80GB / sm_90a ｜ ncu 可用 ｜ pip 直连 PyPI ｜ SGLang pin `0.5.23`（不追新特性）｜ 算子 & AI 编译器 ｜ 4 周完整版。

**因此计划里不再有"待定项"**，只剩一个必须先做的动作：**W0 用 `bw_probe` 把带宽/SM 数实测出来**（§5.2、§10）。数据表数字只作为量级参考，不进报告结论。

---

## 13. 48 小时行动清单

1. `colima start`（或开 Docker Desktop）→ 起 `docker/Dockerfile.cpu` → 跑通 `TRITON_INTERPRET=1` 的 matmul。
2. 校准 GPU：`nvidia-smi` + `bench/bw_probe.py`（实测 HBM 带宽）+ `ncu --version` 且对 toy kernel 采集成功 → 写 `docs/adr/0002-hw-baseline.md`。
3. `git init` 骨架 + pin 版本 + `make test` 绿灯。
4. 只做一件事：**把 decode attention 的 torch 参考 + paged KV 语义写死并测通**（这是后面所有正确性的地基）。
5. 写 `docs/adr/0001-scope.md`，把 §10 的"不做什么"抄进去。
