"""用 libcuda（ctypes）直接加载 cubin 并 launch。

为什么不用 torch 的 CUDA：绕开 torch 版本/CPU-wheel 的问题，让"执行"这一层只依赖驱动。
这也让 GPU 机器上的运行条件最小化：有驱动 + 有 cubin 即可。

⚠️ 这个模块无法在没有 GPU / 没有 libcuda 的机器上验证（Mac 上只能做导入与参数编排检查）。
   首次上卡如有报错，优先核对：符号名（_v2 后缀）、参数类型宽度（CUdeviceptr 是 u64）、
   kernelParams 的指针数组编排。
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import POINTER, byref, c_char_p, c_int, c_size_t, c_uint, c_void_p
from dataclasses import dataclass

__all__ = ["CudaError", "Driver", "DeviceInfo", "load_driver"]

CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


class CudaError(RuntimeError):
    pass


@dataclass
class DeviceInfo:
    name: str
    cc_major: int
    cc_minor: int
    driver_version: int
    total_mem_gb: float

    def __str__(self) -> str:
        return (
            f"{self.name} | sm_{self.cc_major}{self.cc_minor} | "
            f"driver {self.driver_version // 1000}.{(self.driver_version % 1000) // 10} | "
            f"{self.total_mem_gb:.1f} GB"
        )


def load_driver() -> ctypes.CDLL:
    """按平台尝试加载 libcuda。失败时给出可执行的提示，而不是 OSError。"""
    names = ["libcuda.so.1", "libcuda.so", "libcuda.dylib"]
    last: Exception | None = None
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError as e:  # noqa: PERF203
            last = e
    raise CudaError(
        "加载不到 libcuda。\n"
        "  - 本机没有 NVIDIA 驱动/GPU（Mac 上属正常，launcher 只能在 GPU 机器跑）\n"
        "  - 容器里需要 --gpus all 才会挂载驱动库\n"
        f"  - 底层错误：{last}"
    )


class Driver:
    def __init__(self) -> None:
        self.lib = load_driver()
        self._declare_prototypes()

    # ---- 原型声明（类型宽度写错是最常见的崩溃原因）----
    def _declare_prototypes(self) -> None:
        L = self.lib
        L.cuInit.argtypes = [c_uint]
        L.cuInit.restype = c_int
        L.cuDriverGetVersion.argtypes = [POINTER(c_int)]
        L.cuDriverGetVersion.restype = c_int
        L.cuDeviceGet.argtypes = [POINTER(c_int), c_int]
        L.cuDeviceGet.restype = c_int
        L.cuDeviceGetName.argtypes = [c_char_p, c_int, c_int]
        L.cuDeviceGetName.restype = c_int
        L.cuDeviceGetAttribute.argtypes = [POINTER(c_int), c_int, c_int]
        L.cuDeviceGetAttribute.restype = c_int
        L.cuDeviceTotalMem_v2.argtypes = [POINTER(c_size_t), c_int]
        L.cuDeviceTotalMem_v2.restype = c_int
        L.cuCtxCreate_v2.argtypes = [POINTER(c_void_p), c_uint, c_int]
        L.cuCtxCreate_v2.restype = c_int
        L.cuCtxDestroy_v2.argtypes = [c_void_p]
        L.cuCtxDestroy_v2.restype = c_int
        L.cuModuleLoadData.argtypes = [POINTER(c_void_p), c_void_p]
        L.cuModuleLoadData.restype = c_int
        L.cuModuleUnload.argtypes = [c_void_p]
        L.cuModuleUnload.restype = c_int
        L.cuModuleGetFunction.argtypes = [POINTER(c_void_p), c_void_p, c_char_p]
        L.cuModuleGetFunction.restype = c_int
        L.cuMemAlloc_v2.argtypes = [POINTER(ctypes.c_uint64), c_size_t]
        L.cuMemAlloc_v2.restype = c_int
        L.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        L.cuMemFree_v2.restype = c_int
        L.cuMemcpyDtoH_v2.argtypes = [c_void_p, ctypes.c_uint64, c_size_t]
        L.cuMemcpyDtoH_v2.restype = c_int
        L.cuLaunchKernel.argtypes = [
            c_void_p,  # f
            c_uint, c_uint, c_uint,  # grid
            c_uint, c_uint, c_uint,  # block
            c_uint,  # shared mem
            c_void_p,  # stream
            POINTER(c_void_p),  # kernelParams
            POINTER(c_void_p),  # extra
        ]
        L.cuLaunchKernel.restype = c_int
        L.cuCtxSynchronize.argtypes = []
        L.cuCtxSynchronize.restype = c_int
        L.cuGetErrorName.argtypes = [c_int, POINTER(c_char_p)]
        L.cuGetErrorName.restype = c_int
        L.cuGetErrorString.argtypes = [c_int, POINTER(c_char_p)]
        L.cuGetErrorString.restype = c_int

    def _check(self, res: int, what: str) -> None:
        if res == CUDA_SUCCESS:
            return
        name, msg = c_char_p(), c_char_p()
        self.lib.cuGetErrorName(res, byref(name))
        self.lib.cuGetErrorString(res, byref(msg))
        n = name.value.decode() if name.value else str(res)
        m = msg.value.decode() if msg.value else ""
        raise CudaError(f"{what} 失败：{n} ({res}) {m}")

    # ---- 生命周期 ----
    def init(self) -> None:
        self._check(self.lib.cuInit(0), "cuInit")

    def device_info(self) -> DeviceInfo:
        dev = c_int()
        self._check(self.lib.cuDeviceGet(byref(dev), 0), "cuDeviceGet")

        name = ctypes.create_string_buffer(256)
        self._check(self.lib.cuDeviceGetName(name, 256, dev), "cuDeviceGetName")

        major, minor = c_int(), c_int()
        self._check(
            self.lib.cuDeviceGetAttribute(
                byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev
            ),
            "cuDeviceGetAttribute(cc_major)",
        )
        self._check(
            self.lib.cuDeviceGetAttribute(
                byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev
            ),
            "cuDeviceGetAttribute(cc_minor)",
        )

        ver = c_int()
        self._check(self.lib.cuDriverGetVersion(byref(ver)), "cuDriverGetVersion")

        total = c_size_t()
        self._check(self.lib.cuDeviceTotalMem_v2(byref(total), dev), "cuDeviceTotalMem")

        return DeviceInfo(name.value.decode(), major.value, minor.value, ver.value, total.value / 1e9)

    def create_context(self) -> c_void_p:
        ctx = c_void_p()
        self._check(self.lib.cuCtxCreate_v2(byref(ctx), 0, 0), "cuCtxCreate")
        return ctx

    def destroy_context(self, ctx: c_void_p) -> None:
        self._check(self.lib.cuCtxDestroy_v2(ctx), "cuCtxDestroy")

    def load_module(self, image: bytes) -> c_void_p:
        """加载 cubin（也可加载 PTX 文本，驱动会 JIT）。"""
        if not image:
            raise CudaError("模块镜像为空")
        buf = ctypes.create_string_buffer(image, len(image))
        mod = c_void_p()
        self._check(self.lib.cuModuleLoadData(byref(mod), ctypes.cast(buf, c_void_p)), "cuModuleLoadData")
        self._keepalive = buf  # 防止缓冲区被回收
        return mod

    def get_function(self, mod: c_void_p, name: str) -> c_void_p:
        f = c_void_p()
        self._check(self.lib.cuModuleGetFunction(byref(f), mod, name.encode()), f"cuModuleGetFunction({name})")
        return f

    def malloc(self, nbytes: int) -> ctypes.c_uint64:
        d = ctypes.c_uint64()
        self._check(self.lib.cuMemAlloc_v2(byref(d), nbytes), "cuMemAlloc")
        return d

    def free(self, ptr: ctypes.c_uint64) -> None:
        self.lib.cuMemFree_v2(ptr)

    def memcpy_dtoh(self, dst, src_ptr: ctypes.c_uint64, nbytes: int) -> None:
        self._check(self.lib.cuMemcpyDtoH_v2(ctypes.cast(dst, c_void_p), src_ptr, nbytes), "cuMemcpyDtoH")

    def launch(
        self,
        func: c_void_p,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        params: list[tuple[ctypes._SimpleCData, type]],
        shared_mem: int = 0,
    ) -> None:
        """params: [(值对象, ctypes 类型), ...]，顺序必须与 kernel 签名一致。"""
        holders = []
        arr = (c_void_p * len(params))()
        for i, (val, _ty) in enumerate(params):
            holders.append(val)  # 保活
            arr[i] = ctypes.cast(byref(val), c_void_p)
        self._keepalive_params = holders
        self._check(
            self.lib.cuLaunchKernel(
                func,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                shared_mem,
                None,
                arr,
                None,
            ),
            "cuLaunchKernel",
        )

    def sync(self) -> None:
        self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")


def _main() -> int:
    """`python -m minitriton.runtime.launch` 只做环境探测（不 launch）。"""
    try:
        drv = Driver()
        drv.init()
        info = drv.device_info()
    except CudaError as e:
        print(f"[launch] {e}", file=sys.stderr)
        return 2
    print(f"[launch] 设备：{info}")
    print("[launch] 驱动与设备可用。真正的 kernel 执行见 scripts/gpu_hello.py")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
