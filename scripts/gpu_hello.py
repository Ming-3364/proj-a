#!/usr/bin/env python3
"""S1 端到端验证：`.ll → .ptx → .cubin → （GPU）执行 → 对拍`。

**支持 H100(sm_90) / A100(sm_80) 切换**：arch 默认按实际设备自动推断，
产物落在 `build/<arch>/`，两种卡不会互相覆盖。

在 GPU 机器上跑：
    python scripts/gpu_hello.py                 # 自动识别设备
    python scripts/gpu_hello.py --arch sm_80    # A100 强制
    python scripts/gpu_hello.py --inspect       # 只打印设备信息

验收：out[i] == (float)i 对所有 0 <= i < n 成立。
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.hardware import arch_from_cc, resolve  # noqa: E402
from minitriton.runtime import launch, toolchain  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "hello.ll"


def ensure_cubin(arch: str) -> Path:
    """按架构取产物；不存在就现编（cubin 不能跨架构复用）。"""
    cubin = toolchain.default_outdir(arch, ROOT) / "hello.cubin"
    if cubin.exists():
        return cubin
    print(f"[hello] 未找到 {cubin}，按 {arch} 构建…")
    return toolchain.build(FIXTURE, toolchain.default_outdir(arch, ROOT), arch=arch).cubin


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None, help="sm_90(H100) / sm_80(A100)；默认按设备推断")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0, help="0 表示一个线程一个元素")
    ap.add_argument("--inspect", action="store_true", help="只探测设备，不执行")
    args = ap.parse_args(argv)

    try:
        drv = launch.Driver()
        drv.init()
        info = drv.device_info()
    except launch.CudaError as e:
        print(f"[hello] FAIL: {e}", file=sys.stderr)
        return 2

    detected = arch_from_cc(info.cc_major, info.cc_minor)
    print(f"[hello] 设备：{info}")
    print(f"[hello] 检测到 {detected}；档案：{resolve(detected).name if detected in ('sm_80','sm_90') else '未知'}")
    if args.inspect:
        return 0

    arch = args.arch or detected
    if arch not in toolchain.SUPPORTED_ARCHES:
        print(
            f"[hello] FAIL: 不支持的架构 {arch}；本工具支持 {toolchain.SUPPORTED_ARCHES}",
            file=sys.stderr,
        )
        return 2
    if arch != detected:
        print(
            f"[hello] 注意：cubin 按 {arch} 构建，而当前设备是 {detected}。\n"
            "         架构不匹配时驱动会报 CUDA_ERROR_NO_BINARY_FOR_GPU（PTX 才向上兼容）",
            file=sys.stderr,
        )

    n = args.n
    threads = args.threads or n
    cubin = ensure_cubin(arch)
    grid = ((n + threads - 1) // threads, 1, 1)

    ctx = drv.create_context()
    try:
        try:
            mod = drv.load_module(cubin.read_bytes())
        except launch.CudaError as e:
            print(
                f"[hello] FAIL: 加载 cubin 失败：{e}\n"
                f"         十有八九是架构不匹配：cubin 是 {arch}，设备是 {detected}。\n"
                f"         试：python scripts/gpu_hello.py --arch {detected}",
                file=sys.stderr,
            )
            return 3
        fn = drv.get_function(mod, "hello")
        dptr = drv.malloc(n * 4)
        try:
            params = [
                (ctypes.c_uint64(dptr.value), ctypes.c_uint64),  # CUdeviceptr = 64 位
                (ctypes.c_int(n), ctypes.c_int),
            ]
            t0 = time.perf_counter()
            drv.launch(fn, grid, (threads, 1, 1), params)
            drv.sync()
            elapsed_us = (time.perf_counter() - t0) * 1e6

            host = (ctypes.c_float * n)()
            drv.memcpy_dtoh(host, dptr, n * 4)
        finally:
            drv.free(dptr)
    finally:
        drv.destroy_context(ctx)

    bad = [(i, host[i]) for i in range(n) if abs(host[i] - float(i)) > 1e-6]
    print(f"[hello] arch={arch} grid=({grid[0]},1,1) block=({threads},1,1) n={n}")
    print(f"[hello] 前 8 个输出: {[round(host[i], 1) for i in range(min(8, n))]}")
    if bad:
        print(f"[hello] FAIL: {len(bad)} 个元素不匹配，例如 {bad[:3]}", file=sys.stderr)
        return 1
    print(f"[hello] PASS  耗时 {elapsed_us:.1f} µs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
