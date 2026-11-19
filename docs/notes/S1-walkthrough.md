# S1 阅读指南：`.ll → .ptx → .cubin → GPU 执行`

> 目标：让 S1 这 661 行代码**可被逐段读懂**，并说清楚哪些设计是为 S2 埋的伏笔。
> 建议阅读顺序在文末，约 20 分钟。

---

## 1. S1 做了什么 / 没做什么

**做了**：一条把 LLVM IR 变成"能在 H100 上跑出正确结果"的最短路径。
**没做**：任何 IR 设计、任何编译决策。S1 里没有 TTIR、没有 TTGIR、没有 pass。

**为什么先做它**：它不依赖我们的 IR 设计，所以能最先交付；而且它是全链路里唯一会暴露"环境级"失败的部分（PTX 版本、驱动、launcher 参数编排）。

---

## 2. 两条链，别混在一起看

S1 的代码分成**两条完全独立的链**，这是理解它的关键：

```
① 编译链（纯 CPU，Mac 上可跑）
   hello.ll ──llc(NVPTX)──▶ hello.ptx ──ptxas──▶ hello.cubin
   [toolchain.py]                                 产物落在 build/

② 执行链（必须有 GPU + 驱动）
   hello.cubin ──cuModuleLoadData──▶ module
              ──cuModuleGetFunction──▶ fn
              ──cuMemAlloc/cuLaunchKernel/cuMemcpyDtoH──▶ 结果
   [launch.py] 由 [scripts/gpu_hello.py] 串起来
```

两条链之间**只通过一个 `.cubin` 文件通信**。这个边界很值钱：编译链的 bug 在 Mac 上就能复现，执行链的 bug 一定与驱动/参数编排有关——两者不会混在一起。

---

## 3. 文件地图

| 文件 | 行数 | 职责 | 读它的理由 |
|---|---|---|---|
| `tests/fixtures/hello.ll` | 22 | 手写的最小 kernel | 全部语义都在这里，是整条链的"已知正确输入" |
| `minitriton/runtime/toolchain.py` | 149 | llc / ptxas 封装 | 看"工具缺失怎么报错"和"arch 必须一致" |
| `minitriton/runtime/launch.py` | 239 | ctypes + libcuda | 看 ctypes 原型声明与 `kernelParams` 编排 |
| `scripts/gpu_hello.py` | 99 | 端到端验证脚本 | 看执行序列与对拍方式 |
| `tests/test_toolchain.py` | 70 | 编译链回归 | 看"什么算可离线验证" |
| `tests/test_launch.py` | 28 | launcher 的可离线部分 | 看"哪些必须上卡" |
| `Makefile:35` | — | `smoke-cubin` 三路选择 | 看工具链分派的实际策略 |

---

## 4. 逐段读

### 4.1 `hello.ll`：三行关键语义

```llvm
target triple = "nvptx64-nvidia-cuda"        ; 5  目标 GPU
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() ; 7  读硬件线程号（NVVM intrinsic）
define ptx_kernel void @hello(ptr addrspace(1) %out, i32 %n) {  ; 9
```

三个必须记住的点：

1. **`ptx_kernel`（CC 71）**：少了它，`llc` 生成的是 `.visible .func`，**不是 `.entry`，无法 launch**。这是 S1 里被测试抓到的一个真 bug（详见 §5）。
2. **`addrspace(1)`**：1 = global，3 = shared。NVPTX 里指针必须带地址空间，否则驱动侧访问非法。S2 的 lowering 会反复处理这件事。
3. **`read.ptx.sreg.tid.x`**：线程号来自硬件，不是常量。PTX 里会 lower 成 `%tid.x`，测试断言的就是这个。

### 4.2 `toolchain.py`：为什么是两个函数而不是一个

| 位置 | 内容 | 说明 |
|---|---|---|
| `:37 find_llc` | 依次找 `llc, llc-19, ...` | 机器上可能只有版本化名字 |
| `:49 find_ptxas` | 先找 PATH，再找 **triton wheel 自带的 ptxas** | 这是"免装 CUDA toolkit"的关键 |
| `:77 ll_to_ptx` | `-mtriple` + `-mcpu` + `-O` | |
| `:92 ptx_to_cubin` | `ptxas -arch=...` | **arch 必须与 `llc -mcpu` 一致**，否则 PTX 目标与汇编目标不匹配 |
| `:117 build` | 串起来并返回 `BuildResult` | 产物路径 + summary 便于打印 |

设计要点：**错误信息里必须包含下一步**。`ToolchainError` 不是 "FileNotFoundError"，而是"找不到 llc。Mac 上用 make smoke-cubin；容器里 apt-get install llvm"——因为它会出现在别人的终端里。

### 4.3 `launch.py`：ctypes 的三个坑

| 位置 | 内容 | 坑 |
|---|---|---|
| `:68 _declare_prototypes` | 给每个 `cu*` 函数写 `argtypes/restype` | 不写会按 int 传参，64 位指针被截断 → 段错误 |
| `:98` | `cuLaunchKernel.argtypes` 11 个参数 | 顺序极易写错 |
| `:166 load_module` | `cuModuleLoadData` 加载 cubin | **必须保活**：`:173 self._keepalive = buf`，否则 buffer 被 GC，驱动读到野指针 |
| `:192 launch` | 把 `[(值, 类型)]` 编成 `void*[]` | `:206` 同样要保活参数对象 |

`CUdeviceptr` 是 **64 位无符号整数**，不是指针类型——这是 ctypes 里最容易写错的一处（我们的 `:181 malloc` / `:189 memcpy` 都用 `c_uint64`）。

### 4.4 `gpu_hello.py`：执行序列

```
:44  Driver() / init / device_info      # 先探测，失败就退出（exit 2）
:65  create_context()                   # 后面所有 API 都在这个 context 里
:67  load_module(cubin bytes)
:68  get_function(mod, "hello")
:72  params = [ (c_uint64(dptr), ...), (c_int(n), ...) ]   # 严格对应 kernel 签名
:77  launch(fn, grid, block, params)
:88  bad = [...]                         # 逐元素对拍，失败打印前 3 个
```

对拍用的是"逐元素与 `(float)i` 比较"，而不是"跑通就算过"——因为 launch 成功但结果全 0 是最常见的假阳性。

### 4.5 `Makefile:35`：三路分派

本机有 `llc` + `ptxas` → 直接跑；否则有 docker → 进容器；都不行 → 本地跑并让错误信息告诉你缺什么。这解决了一个真实差异：**Mac 有 `llc`（Homebrew LLVM 20，带 NVPTX）但没有 `ptxas`**。

---

## 5. 三个坑（都是被"实跑"暴露的，不是推演出来的）

| 坑 | 怎么发现的 | 教训 |
|---|---|---|
| `.func` 而非 `.entry` | 测试断言 `.entry hello` 失败 | 编译器必须发 `ptx_kernel` CC；否则"编译成功"是假成功 |
| `kernels` 包名撞 HuggingFace 的 PyPI 包 | **只在 H100 机器上** 2 个测试失败 | 环境差异本身要当作被测对象 |
| `bw_probe` 的 L2 报成 51200 MB | 看 GPU 输出时觉得不对 | 单位换算 bug 会让报告里出现荒唐数字 |

---

## 6. 测试策略：为什么是 3 + 2 个测试

| 测试 | 在哪跑 | 守什么 |
|---|---|---|
| `test_ll_compiles_to_ptx_and_cubin` | 容器 | `.entry`、`.target sm_90`、`tid.x`、`.version` 存在 |
| `test_ptx_and_cubin_are_reproducible` | 容器 | 两次构建字节一致 → 否则无法做 golden 与归因 |
| `test_missing_llc_error_is_actionable` | 容器 | 报错含可执行的下一步 |
| `test_launch_module_imports` | 任何机器 | 模块能导入（防空指针/语法问题） |
| `test_missing_driver_error_is_actionable` | 任何机器 | 无驱动时报错可读，而不是裸 OSError |

**可离线验证的边界**：语法、约定、错误信息、可复现性。
**必须上卡的**：真正执行、数值正确性、性能、bank conflict 计数。

---

## 7. 建议阅读顺序（约 20 分钟）

1. `tests/fixtures/hello.ll`（22 行，全部）
2. `scripts/gpu_hello.py`（99 行）——先建立"执行序列"的直觉
3. `minitriton/runtime/toolchain.py`（149 行）——编译链
4. `minitriton/runtime/launch.py` 的 `:45-115`（load_driver + 原型声明 + _check）——最枯燥但最关键
5. `tests/test_toolchain.py`（70 行）——理解"什么算验证过了"
6. 跑一次：`make smoke-cubin`，然后看 `build/hello.ptx` 的前 20 行

---

## 8. 这些设计如何影响 S2

| S1 的产物 | S2 会怎么用 |
|---|---|
| `ll_to_ptx` / `ptx_to_cubin` | 原样复用，只是输入从手写 `.ll` 变成 lowering 生成的 `.ll` |
| `ptx_kernel` 约定 | S2 的 lowering 必须无条件遵守（这是易错点 #1） |
| `addrspace(1)` | S2 要为每个 buffer 决定地址空间（global=1，shared=3） |
| `kernelParams` 编排 | S2 的 kernel 参数更多（M/N/K/stride/ptr），编排要泛化，不能写死两个 |
| `device_info` + 对拍 | S2 直接复用 `gpu_hello.py` 的骨架，换成"与 torch 对拍 + 多 shape" |
