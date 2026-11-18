"""工具链调用：`.ll --llc--> .ptx --ptxas--> .cubin`。

为什么单独成模块：这是**唯一不依赖我们 IR 设计**的一层，所以能最先交付、也最先暴露环境问题
（LLVM 版本、NVPTX 后端、PTX 版本与驱动的兼容性）。

两个工具都是纯 CPU 程序，因此整条链在 Mac 容器里就能跑完，只有"执行"需要 GPU。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ToolchainError",
    "find_llc",
    "find_ptxas",
    "ll_to_ptx",
    "ptx_to_cubin",
    "build",
    "DEFAULT_TRIPLE",
    "DEFAULT_CPU",
]

DEFAULT_TRIPLE = "nvptx64-nvidia-cuda"
DEFAULT_CPU = "sm_90"  # H100 PCIe


class ToolchainError(RuntimeError):
    """工具缺失或调用失败。消息里必须包含下一步该怎么做。"""


def find_llc() -> str:
    for name in ("llc", "llc-19", "llc-18", "llc-17", "llc-16", "llc-15", "llc-14"):
        p = shutil.which(name)
        if p:
            return p
    raise ToolchainError(
        "找不到 llc。\n"
        "  - Mac 上：llc 在容器里，用 `make smoke-cubin`（会自动走 docker）\n"
        "  - 容器/GPU 机上：apt-get install llvm（Debian 的 llvm 自带 NVPTX 后端）"
    )


def find_ptxas() -> str:
    p = shutil.which("ptxas")
    if p:
        return p
    # triton 的 wheel 自带 ptxas，不必单独装 CUDA toolkit
    try:
        import triton  # noqa: PLC0415

        cand = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
        if cand.exists():
            return str(cand)
    except Exception:
        pass
    raise ToolchainError(
        "找不到 ptxas。\n"
        "  - 装 CUDA toolkit 后 ptxas 在 $CUDA_HOME/bin\n"
        "  - 或 pip install triton（wheel 自带 ptxas，本项目默认走这条）"
    )


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise ToolchainError(
            "命令失败：" + " ".join(cmd) + "\n--- stderr ---\n" + (r.stderr or r.stdout)[:2000]
        )


def ll_to_ptx(
    ll: Path,
    ptx: Path,
    cpu: str = DEFAULT_CPU,
    triple: str = DEFAULT_TRIPLE,
    opt: str = "3",
) -> Path:
    ll, ptx = Path(ll), Path(ptx)
    if not ll.exists():
        raise ToolchainError(f"输入不存在：{ll}")
    ptx.parent.mkdir(parents=True, exist_ok=True)
    _run([find_llc(), f"-mtriple={triple}", f"-mcpu={cpu}", f"-O{opt}", str(ll), "-o", str(ptx)])
    return ptx


def ptx_to_cubin(ptx: Path, cubin: Path, arch: str = DEFAULT_CPU) -> Path:
    ptx = Path(ptx)
    cubin = Path(cubin)
    cubin.parent.mkdir(parents=True, exist_ok=True)
    # 注意：ptxas 的 arch 必须与 llc 的 -mcpu 一致，否则 PTX 版本与目标不匹配
    _run([find_ptxas(), f"-arch={arch}", str(ptx), "-o", str(cubin)])
    if cubin.stat().st_size == 0:
        raise ToolchainError(f"ptxas 产出为空：{cubin}")
    return cubin


@dataclass
class BuildResult:
    ll: Path
    ptx: Path
    cubin: Path
    cpu: str

    def summary(self) -> str:
        return (
            f"ll={self.ll.name}  ptx={self.ptx.name} ({self.ptx.stat().st_size}B)  "
            f"cubin={self.cubin.name} ({self.cubin.stat().st_size}B)  cpu={self.cpu}"
        )


def build(ll: Path, outdir: Path, cpu: str = DEFAULT_CPU, opt: str = "3") -> BuildResult:
    ll, outdir = Path(ll), Path(outdir)
    stem = ll.stem
    ptx = ll_to_ptx(ll, outdir / f"{stem}.ptx", cpu=cpu, opt=opt)
    cubin = ptx_to_cubin(ptx, outdir / f"{stem}.cubin", arch=cpu)
    return BuildResult(ll=ll, ptx=ptx, cubin=cubin, cpu=cpu)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=".ll -> .ptx -> .cubin（不需要 GPU）")
    ap.add_argument("ll", type=Path, help="输入的 .ll 文件")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("build"))
    ap.add_argument("--cpu", default=DEFAULT_CPU, help=f"默认 {DEFAULT_CPU}")
    ap.add_argument("--opt", default="3")
    args = ap.parse_args(argv)

    try:
        res = build(args.ll, args.outdir, cpu=args.cpu, opt=args.opt)
    except ToolchainError as e:
        print(f"[toolchain] {e}", file=sys.stderr)
        return 2

    print(res.summary())
    head = res.ptx.read_text(encoding="utf-8", errors="replace").splitlines()[:12]
    print("--- PTX 头部 ---")
    for line in head:
        print("  " + line)
    print(f"下一步：把 {res.cubin} 拷到 H100，用 libcuda 加载并 launch（S1 的另一半）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
