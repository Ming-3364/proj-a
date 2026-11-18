"""实测 HBM 带宽 + 打印设备指纹。**W0 的硬性产出。**

为什么不用数据表：H100 有 SXM / PCIe / NVL 三个版本，带宽 3.35 / 2.0 / 3.9 TB/s、
SM 数 132 / 114，网上的教程几乎都默认 SXM。照抄会让所有 roofline 结论偏移 1.7 倍。
所以：**先测，再建模**。测出来的数字写进 docs/adr/0002-hw-baseline.md。

用法（GPU 机器上）：
    python bench/bw_probe.py                       # 默认 2 GiB / 20 次，取中位数
    python bench/bw_probe.py --size-gb 8 --iters 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def device_fingerprint() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False}
    p = torch.cuda.get_device_properties(0)
    info = {
        "cuda": True,
        "name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "total_memory_gib": round(p.total_memory / 1024**3, 2),
    }
    # 字段名随 torch 版本变动，全部走 getattr 兜底
    unit_div = {"l2_cache_mb": 1024**2, "smem_per_sm_kb": 1024, "smem_per_block_optin_kb": 1024}
    for key, attr in [
        ("l2_cache_mb", "L2_cache_size"),
        ("smem_per_sm_kb", "shared_memory_per_multiprocessor"),
        ("smem_per_block_optin_kb", "shared_memory_per_block_optin"),
        ("max_threads_per_sm", "max_threads_per_multi_processor"),
    ]:
        val = getattr(p, attr, None)
        if val:
            info[key] = round(val / unit_div[key], 1) if key in unit_div else val
    return info


def _timeit(fn, iters: int) -> float:
    """返回中位耗时（秒）。用 cuda events，而不是 wall clock + synchronize。"""
    import torch

    for _ in range(3):  # warmup
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    return statistics.median(times)


def measure(size_gb: float, iters: int) -> dict:
    import torch

    n = int(size_gb * 1024**3 // 2)  # fp16：每元素 2 字节
    src = torch.empty(n, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)

    # 1) copy：读 + 写。最接近「kernel 每读 1 字节就产出 1 字节」的场景
    t_copy = _timeit(lambda: dst.copy_(src), iters)
    bw_copy = (2 * n * 2) / t_copy / 1e12

    # 2) reduce：纯读。作为「只读 KV cache」的上界参考
    t_read = _timeit(lambda: src.sum(), iters)
    bw_read = (n * 2) / t_read / 1e12

    del src, dst
    torch.cuda.empty_cache()

    return {
        "size_gb": size_gb,
        "iters": iters,
        "copy_sec": round(t_copy, 6),
        "copy_TBps": round(bw_copy, 3),
        "read_sec": round(t_read, 6),
        "read_TBps": round(bw_read, 3),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-gb", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", type=str, default=None, help="JSON 输出路径")
    args = ap.parse_args(argv)

    info = device_fingerprint()
    if not info.get("cuda"):
        print("SKIP: 无 CUDA 设备（本机是 Apple Silicon，符合预期）")
        return 0

    print(json.dumps(info, indent=2, ensure_ascii=False))
    res = measure(args.size_gb, args.iters)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(
        f"\n>>> 把 BW_MEASURED = {res['copy_TBps']:.2f} TB/s (copy) / "
        f"{res['read_TBps']:.2f} TB/s (read) 写进 docs/adr/0002-hw-baseline.md"
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"device": info, "bandwidth": res}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
