"""让 `pytest` 在仓库根目录下能 import 到 refs/ 与 kernels/（容器里用）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
