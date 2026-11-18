#!/usr/bin/env python3
"""S1 端到端验证：`.ll → .ptx → .cubin → （GPU）执行 → 对拍`。

在 GPU 机器上跑：
    python scripts/gpu_hello.py            # 默认 n=32
    python scripts/gpu_hello.py --n 1024 --threads 256
    python scripts/gpu_hello.py --inspect  # 只打印设备信息，不执行

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

from minitriton.runtime import launch, toolchain  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "hello.ll"


def ensure_cubin(cubin: Path) -> Path:
    if cubin.exists():
        return cubin
    print(f"[hello] 未找到 {cubin}，先构建…")
    return toolchain.build(FIXTURE, cubin.parent, cpu="sm_90").cubin


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubin", type=Path, default=ROOT / "build" / "hello.cubin")
    ap.add_argument("--n", type=int, default=32, help="元素个数")
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

    print(f"[hello] 设备：{info}")
    if args.inspect:
        return 0
    if (info.cc_major, info.cc_minor) < (9, 0):
        print(
            f"[hello] 注意：设备是 sm_{info.cc_major}{info.cc_minor}，"
            "而 cubin 按 sm_90 构建，可能无法加载"
        )

    n = args.n
    threads = args.threads or n
    cubin = ensure_cubin(args.cubin)
    grid = ((n + threads - 1) // threads, 1, 1)

    ctx = drv.create_context()
    try:
        mod = drv.load_module(cubin.read_bytes())
        fn = drv.get_function(mod, "hello")
        dptr = drv.malloc(n * 4)  # float32 * n
        try:
            # kernel 签名: hello(float* out, int n)
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
    print(f"[hello] grid=({grid[0]},1,1) block=({threads},1,1) n={n}")
    print(f"[hello] 前 8 个输出: {[round(host[i], 1) for i in range(min(8, n))]}")
    if bad:
        print(f"[hello] FAIL: {len(bad)} 个元素不匹配，例如 {bad[:3]}", file=sys.stderr)
        return 1
    print(f"[hello] PASS  耗时 {elapsed_us:.1f} µs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
