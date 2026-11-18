"""S1 验收：`.ll → .ptx → .cubin` 能在本机跑通（不需要 GPU）。

这两个测试的存在意义：把"工具链可用"从一句口头承诺变成一个每次 `make test` 都会检查的事实。
没有 llc 的环境（比如没进容器的 Mac）会 skip，并说明该怎么做。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from minitriton.runtime import toolchain as tc

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hello.ll"


def _require_tools():
    try:
        llc = tc.find_llc()
        ptxas = tc.find_ptxas()
    except tc.ToolchainError as e:
        raise unittest.SkipTest(f"工具链不可用（{str(e).splitlines()[0]}）；在容器内跑即可")
    return llc, ptxas


def test_ll_compiles_to_ptx_and_cubin():
    _require_tools()
    with tempfile.TemporaryDirectory() as d:
        res = tc.build(FIXTURE, Path(d), cpu="sm_90")

        assert res.ptx.exists() and res.ptx.stat().st_size > 0
        assert res.cubin.exists() and res.cubin.stat().st_size > 0, "ptxas 应产出非空 cubin"

        ptx = res.ptx.read_text()
        # 目标与入口：证明是给 sm_90 编的、函数名没被改写
        assert ".target sm_90" in ptx, ptx[:200]
        assert ".entry hello" in ptx, "kernel 入口名应保留"
        # NVVM intrinsic 生效：线程 id 来自硬件寄存器，而不是常量
        assert "tid.x" in ptx, "read.ptx.sreg.tid.x 应被 lower 成 %tid.x"
        # PTX 版本写在文件头，驱动兼容性看它
        assert any(l.startswith(".version") for l in ptx.splitlines()[:10]), "PTX 头部应有 .version"


def test_ptx_and_cubin_are_reproducible():
    """同样的输入两次构建应产出字节一致的 cubin（否则无法做 golden 与归因）。"""
    _require_tools()
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a = tc.build(FIXTURE, Path(d1), cpu="sm_90")
        b = tc.build(FIXTURE, Path(d2), cpu="sm_90")
        assert a.ptx.read_text() == b.ptx.read_text()
        assert a.cubin.read_bytes() == b.cubin.read_bytes()


def test_missing_llc_error_is_actionable():
    """工具缺失时的报错必须告诉人下一步怎么做，而不是抛 FileNotFoundError。"""
    orig = tc.find_llc
    tc.find_llc = lambda: (_ for _ in ()).throw(tc.ToolchainError("找不到 llc。\n  - 用 make smoke-cubin"))
    try:
        with tempfile.TemporaryDirectory() as d:
            try:
                tc.build(FIXTURE, Path(d))
            except tc.ToolchainError as e:
                msg = str(e)
                assert "llc" in msg
                assert "make smoke-cubin" in msg, "报错里应给出可执行的下一步"
            else:
                raise AssertionError("应当抛出 ToolchainError")
    finally:
        tc.find_llc = orig
