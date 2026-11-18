# minitriton 语言子集（v0 契约）

> 这份文档是**契约**，不是愿望清单。凡不在此列的写法，编译器必须抛 `UnsupportedError` 并说明原因，**绝不静默降级**。

## 1. kernel 形状

```python
import triton, triton.language as tl

@triton.jit
def kernel_name(arg_ptr, ..., META: tl.constexpr):
    ...
```

- 必须带 `@triton.jit` 装饰器；一个文件可以有多个 kernel，用 `--kernel` 指定。
- 参数分两类：**指针参数**（默认 `ptr<f32>`）与 **`tl.constexpr` 参数**（编译期整数，由 `--meta NAME=VALUE` 提供）。
- 不支持 return 值；kernel 通过 `tl.store` 写输出。

## 2. 支持的语法（v0）

| 类别 | 支持 | 备注 |
|---|---|---|
| 表达式 | `+ - * /`、比较 `< <= > >= == !=`、括号 | 张量之间或张量与标量 |
| 赋值 | `x = expr`（单目标） | 多目标赋值不支持 |
| 调用 | `tl.program_id`、`tl.arange`、`tl.load`、`tl.store`、`tl.zeros` | 见下 |
| 归约 | `tl.max(x, axis=0)`、`tl.sum(x, axis=0)`、`tl.exp` | v0 仅 1D、axis=0 |
| 控制流 | 无（kernel 内不允许 if/for，v0） | 循环由 BLOCK 分块在 IR 层表达 |

## 3. `tl.*` 语义（v0 的唯一实现口径）

| 调用 | 结果 |
|---|---|
| `tl.program_id(axis)` | `scalar<i32>`，axis 为编译期常量 |
| `tl.arange(0, BLOCK)` | `tensor<BLOCK:f32>`，起点恒为 0 |
| `ptr + offs` | `addr`（指针张量），供 load/store 消费 |
| `tl.load(addr, mask=..., other=...)` | `tensor<BLOCK:f32>`，mask/other 可省 |
| `tl.store(addr, value, mask=...)` | 无返回值（语句） |
| `tl.zeros((BLOCK,), tl.float32)` | `tensor<BLOCK:f32>` |
| `tl.max/sum(x, axis=0)` | 标量归约，形状降一维 |
| `tl.exp(x)` | 逐元素 |

## 4. 明确不支持（v0 会报错，不是遗漏）

`tl.trans`、`tl.dot`、atomic、`make_block_ptr`、`make_tensor_descriptor`(TMA)、`inline_asm`、
warp 特化、多返回值、数据相关的 if/while、嵌套函数、`for` 循环、跨 kernel 调用。

> `tl.dot` 与 `for` 会在 **W2（GEMM）** 引入；届时按本文件的格式追加条目，并同步更新 §3。

## 5. 错误约定

`UnsupportedError` 必须包含三要素：**位置（行号）+ 不支持的构造 + 建议替代写法**。
理由：报错信息本身就是交付物——它证明编译器知道自己的边界在哪。
