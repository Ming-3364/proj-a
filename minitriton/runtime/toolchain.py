"""工具链调用：`.ll --llc--> .ptx --ptxas--> .cubin`。

**架构是显式参数**（`arch="sm_90"` / `"sm_80"`），因为它决定产物能否在目标卡上跑：
cubin 与架构绑定，PTX 才向上兼容（sm_80 的 PTX 能被 sm_90 驱动 JIT，反之不行）。

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
    "default_outdir",
    "DEFAULT_TRIPLE",
    "DEFAULT_ARCH",
    "SUPPORTED_ARCHES",
]

DEFAULT_TRIPLE = "nvptx64-nvidia-cuda"
DEFAULT_ARCH = "sm_90"  # H100
# A100 = sm_80；两者产物不能互换
SUPPORTED_ARCHES = ("sm_80", "sm_90", "sm_90a")


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


def _check_arch(arch: str) -> str:
    if arch not in SUPPORTED_ARCHES:
        raise ToolchainError(
            f"不支持的 arch={arch!r}；可选 {', '.join(SUPPORTED_ARCHES)}。\n"
            "  H100 = sm_90，A100 = sm_80（cubin 不能跨架构复用）"
        )
    return arch


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise ToolchainError(
            "命令失败：" + " ".join(cmd) + "\n--- stderr ---\n" + (r.stderr or r.stdout)[:2000]
        )


def ll_to_ptx(
    ll: Path,
    ptx: Path,
    arch: str = DEFAULT_ARCH,
    triple: str = DEFAULT_TRIPLE,
    opt: str = "3",
) -> Path:
    arch = _check_arch(arch)
    ll, ptx = Path(ll), Path(ptx)
    if not ll.exists():
        raise ToolchainError(f"输入不存在：{ll}")
    ptx.parent.mkdir(parents=True, exist_ok=True)
    _run([find_llc(), f"-mtriple={triple}", f"-mcpu={arch}", f"-O{opt}", str(ll), "-o", str(ptx)])
    return ptx


def ptx_to_cubin(ptx: Path, cubin: Path, arch: str = DEFAULT_ARCH) -> Path:
    arch = _check_arch(arch)
    ptx = Path(ptx)
    cubin = Path(cubin)
    cubin.parent.mkdir(parents=True, exist_ok=True)
    # ptxas 的 arch 必须与 llc 的 -mcpu 一致，否则 PTX 目标与汇编目标不匹配
    _run([find_ptxas(), f"-arch={arch}", str(ptx), "-o", str(cubin)])
    if cubin.stat().st_size == 0:
        raise ToolchainError(f"ptxas 产出为空：{cubin}")
    return cubin


def default_outdir(arch: str = DEFAULT_ARCH, root: Path | None = None) -> Path:
    """产物按架构分目录：build/sm_90/、build/sm_80/。避免两种卡互相覆盖。"""
    return (root or Path("build")) / arch


@dataclass
class BuildResult:
    ll: Path
    ptx: Path
    cubin: Path
    arch: str

    def summary(self) -> str:
        return (
            f"ll={self.ll.name}  ptx={self.ptx.name} ({self.ptx.stat().st_size}B)  "
            f"cubin={self.cubin.name} ({self.cubin.stat().st_size}B)  arch={self.arch}"
        )


def build(ll: Path, outdir: Path | None = None, arch: str = DEFAULT_ARCH, opt: str = "3") -> BuildResult:
    ll = Path(ll)
    outdir = Path(outdir) if outdir is not None else default_outdir(arch)
    stem = ll.stem
    ptx = ll_to_ptx(ll, outdir / f"{stem}.ptx", arch=arch, opt=opt)
    cubin = ptx_to_cubin(ptx, outdir / f"{stem}.cubin", arch=arch)
    return BuildResult(ll=ll, ptx=ptx, cubin=cubin, arch=arch)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=".ll -> .ptx -> .cubin（不需要 GPU）")
    ap.add_argument("ll", type=Path, help="输入的 .ll 文件")
    ap.add_argument("-o", "--outdir", type=Path, default=None, help="默认 build/<arch>/")
    ap.add_argument("--arch", "--cpu", dest="arch", default=DEFAULT_ARCH, help=f"默认 {DEFAULT_ARCH}")
    ap.add_argument("--opt", default="3")
    args = ap.parse_args(argv)

    try:
        res = build(args.ll, args.outdir, arch=args.arch, opt=args.opt)
    except ToolchainError as e:
        print(f"[toolchain] {e}", file=sys.stderr)
        return 2

    print(res.summary())
    head = res.ptx.read_text(encoding="utf-8", errors="replace").splitlines()[:12]
    print("--- PTX 头部 ---")
    for line in head:
        print("  " + line)
    print(f"下一步：把 {res.cubin} 拷到目标卡（{res.arch}），用 libcuda 加载并 launch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
