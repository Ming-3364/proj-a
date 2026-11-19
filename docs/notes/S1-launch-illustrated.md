# 图解 `launch.py`：一层 FFI，四组动作

> 读法：每张图只增加一层信息。看完图 0 到图 10，这个文件就没有黑箱了。

---

## 图 0：全景位置

```
┌─────────────────── Python 进程 ────────────────────┐
│  gpu_hello.py        编排"先谁后谁"                  │
│        │                                           │
│        ▼                                           │
│  launch.py     Driver 类的 12 个方法   ◀── 本文主角  │
│        │  ◀── ctypes 边界：FFI（外部函数接口）        │
├────────┼───────────────────────────────────────────┤
│        ▼                                           │
│  libcuda.so.1     C ABI，函数名 cu*                 │
│        │                                           │
│        ▼                                           │
│  内核态驱动  ──▶  H100 硬件                          │
└────────────────────────────────────────────────────┘
```

`launch.py` 里**没有任何计算逻辑**。它只做三件事：管理句柄、适配 ABI、翻译错误。

---

## 图 1：十二个方法分组

```
第一组  装载与声明        load_driver() / __init__() / _declare_prototypes()
第二组  错误翻译          _check()
第三组  查询              device_info()
第四组  资源与执行        create_context() destroy_context()
                         load_module() get_function()
                         malloc() free() memcpy_dtoh()
                         launch() sync()
入口                     _main()   ← 只探测，不执行
```

---

## 图 2：`__init__` 只做两件事

```python
class Driver:
    def __init__(self):
        self.lib = load_driver()        # ① dlopen，拿到一张符号表
        self._declare_prototypes()      # ② 给每个符号贴类型标签
```

```
Driver()
 ├── self.lib = <CDLL 'libcuda.so.1'>    ← dlopen 的产物
 │        └─ .cuInit / .cuMemAlloc_v2 / ...   ← 符号按需解析
 └── 类型标签表（argtypes / restype）
```

**dlopen 不加载函数体**，只是把 `.so` 映射进进程；符号在第一次被访问时解析。

---

## 图 3：为什么必须写 `argtypes` / `restype`

```
参数侧：宽度由"你传的 ctypes 对象"决定
  c_int(n)        → 4 字节   ✅ 用来传 int
  c_int(ptr)      → 4 字节   ❌ 地址被砍掉高 32 位（静默）
  c_uint64(ptr)   → 8 字节   ✅ 用来传 CUdeviceptr

返回侧：不写 restype 就默认 int（32 位）
  ┌──────────────────────────────┐
  │ 驱动返回的真实句柄             │
  │   0x00007f3c_12345678         │
  │ 你拿到的（被截断）             │
  │   0x00000000_12345678         │ ← 高位丢失，后续使用=野指针
  └──────────────────────────────┘
```

这就是 `_declare_prototypes()` 存在的唯一理由：**把 ABI 说清楚，否则 Python 猜错**。

---

## 图 4：三种"数字"，别混

| 东西 | ctypes 类型 | 含义 | 谁分配 |
|---|---|---|---|
| `ctx` / `mod` / `fn` | `c_void_p` | **不透明句柄**（内容是驱动私有的） | 驱动 |
| `dptr` | `c_uint64` | **设备地址**（是一个数值，不是指针） | 驱动（`cuMemAlloc` 填入） |
| `nbytes` | `c_size_t` | 字节数 | 你 |

所以代码里到处是 `POINTER(c_void_p)`（要写回句柄）和 `POINTER(c_uint64)`（要写回地址）。

---

## 图 5：`load_module` 的数据流与 GC 陷阱

```python
buf = ctypes.create_string_buffer(image, len(image))
self.lib.cuModuleLoadData(byref(mod), ctypes.cast(buf, c_void_p))
self._keepalive = buf        # ★
```

```
cubin 的 bytes（Python 对象）
      │ create_string_buffer
      ▼
buf = [ 7f 45 4c 46 ... ]        ← 一块 Python 管理的堆内存，ELF magic
      │ cast → void*
      ▼
cuModuleLoadData(&mod, buf)      驱动在 C 侧读这块内存，解析 ELF
      │
      ▼
self._keepalive = buf            ← 保住引用，否则 buf 被 GC
```

**GC 视角**：Python 不知道 C 侧还指着 `buf`。引用归零 → 内存回收 → 驱动手里变悬垂指针。崩溃可能延迟到下一次调用，所以极难排查。

---

## 图 6：`launch` 的参数编排（最容易错的地方）

```python
arr = (c_void_p * len(params))()                 # void* 数组
for i, (val, _ty) in enumerate(params):
    arr[i] = ctypes.cast(byref(val), c_void_p)   # 存"指向值的指针"
```

hello 的实际内存布局：

```
params = [(c_uint64(dptr), ...), (c_int(n), ...)]

  val0: 0x00007f3c_2a001000   8 字节   ← 设备地址
  val1: 32                    4 字节   ← n

  arr: [ &val0 , &val1 ]      两个 void*
          │        │
          └────────┴──▶ cuLaunchKernel(fn, ..., arr, ...)
                                   │
                        驱动按 PTX 签名逐个读：
                        .param .u64 hello_param_0  ← 从 arr[0] 读 8 字节
                        .param .u32 hello_param_1  ← 从 arr[1] 读 4 字节
```

**要点**：数组元素是「指针」，不是「值」。参数宽度必须与 PTX 里的 `.param` 严格一致。

---

## 图 7：一次调用的时序，注意那个异步点

```
Python              ctypes/libcuda                    GPU
  │  create_context      │                              │
  │──cuCtxCreate────────▶│  建上下文                     │
  │                      │                              │
  │  load_module         │                              │
  │──cuModuleLoadData───▶│  解析 ELF                     │
  │                      │                              │
  │  launch              │                              │
  │──cuLaunchKernel─────▶│  打包参数 → 入队 ──────────────▶ 开始跑
  │  ★ 立即返回          │                              │
  │                      │                              │
  │  sync                │                              │
  │──cuCtxSynchronize───▶│  阻塞等待 ────────────────────▶ 完成
  │                      │                              │
  │  memcpy_dtoh         │                              │
  │──cuMemcpyDtoH───────▶│  拷回 ◀────────────────────────┘
```

因为 `launch` 是入队即返回，所以**不 `sync` 就可能读到旧数据**。

顺带：我们测的 68.4 µs 是「launch + 同步等待」的往返，包含驱动开销，不是纯 kernel 时间。

---

## 图 8：错误是怎么被翻译的

```
cu* 函数返回 CUresult（一个 int）
        │ != 0
        ▼
_check(res, "cuModuleLoadData")
        ├─ cuGetErrorName(res)   → "CUDA_ERROR_INVALID_IMAGE"
        └─ cuGetErrorString(res) → "invalid device function"
        ▼
raise CudaError("cuModuleLoadData 失败：CUDA_ERROR_INVALID_IMAGE (200) invalid device function")
```

价值：报错**自带"哪个调用失败了"的上下文**，而不是一个孤零零的数字。

---

## 图 9：一页纸总结

```
Driver
 ├── lib                 ← dlopen 的符号表（第一组）
 ├── argtypes/restype    ← ABI 标签（第一组）
 ├── _check              ← CUresult → 异常（第二组）
 ├── device_info         ← 只读查询（第三组）
 └── 句柄流（第四组）
       cuCtxCreate            → ctx
         cuModuleLoadData     → mod        （+ keepalive buffer）
           cuModuleGetFunction→ fn
         cuMemAlloc           → dptr(u64)
         cuLaunchKernel(fn, grid, block, smem, stream, params)
         cuCtxSynchronize
         cuMemcpyDtoH(dptr → host)
```

**心智模型**：

> `launch.py` = **句柄管理器 + ABI 适配器 + 错误翻译器**。
> 它不含计算，也不含调度策略；调度在 `gpu_hello.py`，计算在 GPU 上。

---

## 附录：三个坑的对照表

| 坑 | 症状 | 机制 | 位置 |
|---|---|---|---|
| 宽度错 | 段错误 / 结果乱 | 指针被截成 32 位 | 图 3 |
| GC 回收 | 随机崩溃（延迟） | C 侧持有 Python 内存 | 图 5 |
| 忘了 sync | 读到 0 或旧数据 | launch 是异步的 | 图 7 |
