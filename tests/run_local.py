"""零依赖测试 runner：Mac 上没有 pytest 也能跑。

约定：测试文件里写成普通的 `def test_xxx():` 函数（无参数、用 assert），
这样容器里装了 pytest 后 `pytest tests/` 能直接收集同一批函数。
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    sys.path.insert(0, str(ROOT))
    files = sorted(p for p in (ROOT / "tests").glob("test_*.py"))
    if not files:
        print("no tests found")
        return 1

    passed, failed, skipped = 0, 0, 0
    failures = []

    for f in files:
        try:
            mod = _load(f)
        except Exception:
            failed += 1
            failures.append((f.name, "<import>", traceback.format_exc()))
            continue

        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
            except unittest.SkipTest as e:
                skipped += 1
                print(f"SKIP {f.name}::{name}  ({e})")
            except AssertionError:
                failed += 1
                failures.append((f.name, name, traceback.format_exc()))
                print(f"FAIL {f.name}::{name}")
            except Exception:
                failed += 1
                failures.append((f.name, name, traceback.format_exc()))
                print(f"ERROR {f.name}::{name}")
            else:
                passed += 1
                print(f"ok   {f.name}::{name}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    if failures:
        print("\n" + "=" * 70)
        for fname, tname, tb in failures:
            print(f"\n### {fname}::{tname}\n{tb}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
