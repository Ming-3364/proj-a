"""MLIR 插件 spike 的回归测试。

守住一件事：stock `mlir-opt` 能加载我们 out-of-tree 的方言插件。
这是整套 MLIR 方案的前提——它挂了，后面所有 dialect/pass 都无从谈起。

不需要 GPU；需要 brew 的 llvm（含 mlir-opt）。没有就 skip。
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MLIR = ROOT / "mlir"


def _mlir_opt() -> str:
    for cand in ("mlir-opt", "/opt/homebrew/opt/llvm@20/bin/mlir-opt"):
        p = shutil.which(cand) or (cand if Path(cand).exists() else None)
        if p:
            return p
    raise unittest.SkipTest("找不到 mlir-opt（brew install llvm 后可用）")


def test_plugin_builds_and_loads():
    opt = _mlir_opt()
    if not (MLIR / "Makefile").exists():
        raise unittest.SkipTest("mlir/ 目录不存在")

    build = subprocess.run(
        ["make", "-s", "all"], cwd=MLIR, capture_output=True, text=True
    )
    assert build.returncode == 0, f"插件构建失败:\n{build.stderr[-2000:]}"

    dylib = MLIR / "build" / "libminittir.dylib"
    assert dylib.exists(), "应有 build/libminittir.dylib"

    run = subprocess.run(
        [opt, f"--load-dialect-plugin={dylib}", str(MLIR / "tests" / "spike.mlir")],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, f"mlir-opt 加载失败:\n{run.stderr[-2000:]}"
    assert "ttir.add" in run.stdout, run.stdout
    # 方言解析成功意味着 ODS 的 assemblyFormat 生效
    assert "tensor<16xf32>" in run.stdout
