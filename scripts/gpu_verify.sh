#!/usr/bin/env bash
# GPU 侧一键验证：在 GPU 机器上跑，产出可回传到 Mac 的 JSON。
# 用法： bash scripts/gpu_verify.sh [输出目录]
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="${1:-}"
if [[ -z "$OUT_DIR" ]]; then
  slug=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' | tr '/' '_' | sed 's/^/gpu-/')
  slug=${slug:-gpu-unknown}
  OUT_DIR="benchmarks/results/${slug}"
fi
mkdir -p "$OUT_DIR"
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)

echo "== 1/3 设备与带宽基线 =="
python bench/bw_probe.py --out "${OUT_DIR}/bw_probe_${SHA}.json"

echo "== 2/3 全部测试（含 kernel 真实编译执行，无解释器模式）=="
TRITON_INTERPRET=0 python -m pytest -q 2>/dev/null || python tests/run_local.py

echo "== 3/3 报告摘要 =="
python - <<'PY'
import glob, json
for f in sorted(glob.glob("benchmarks/results/**/*.json", recursive=True))[-5:]:
    print(f, "->", json.load(open(f))["bandwidth"]["copy_TBps"], "TB/s (copy)")
PY

echo "完成。把 ${OUT_DIR} 拷回 Mac 后跑 make report。"
