"""launcher 的"可离线验证"部分。

真正 launch 需要 GPU；这里只守住两点：
  1. 模块能导入（没有语法/类型注解错误）；
  2. 找不到 libcuda 时报错是**可执行的**，不是裸 OSError。
"""

from __future__ import annotations

from minitriton.runtime import launch


def test_launch_module_imports():
    assert hasattr(launch, "Driver")
    assert hasattr(launch.Driver, "launch")
    assert hasattr(launch.Driver, "load_module")


def test_missing_driver_error_is_actionable():
    try:
        launch.load_driver()
    except launch.CudaError as e:
        msg = str(e)
        assert "libcuda" in msg
        assert "GPU" in msg or "驱动" in msg, "报错应说明为什么不可用"
    else:
        # 有驱动的机器（GPU 服务器）走到这里也算通过
        pass
