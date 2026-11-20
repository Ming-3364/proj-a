# 本地（Mac）可跑的目标一律不依赖 GPU 与 triton
# 自动挑选「能 import torch」的解释器：PATH 里可能有多个 python3（系统版通常没有 torch）
ARCH ?= sm_90   # H100=sm_90 / A100=sm_80
PY ?= $(shell for p in python3 $$HOME/miniconda3/bin/python3 /opt/homebrew/bin/python3; do \
	if $$p -c "import torch" >/dev/null 2>&1; then echo $$p; exit 0; fi; done; echo python3)

.PHONY: help test dev-shell bw gpu-verify smoke smoke-cubin mlir-spike clean

help:
	@echo "make doctor      - 打印解释器/依赖体检（PATH 里可能有多个 python3）"
	@echo "make test        - 本地纯 CPU 测试（参考实现 / 语义），零依赖"
	@echo "make smoke       - 本地冒烟：参考实现自检 + 打印版本"
	@echo "make smoke-cubin - S1：.ll -> .ptx -> .cubin（ARCH=sm_90|sm_80 切换 H100/A100）"
	@echo "make mlir-spike  - MLIR 插件：mlir-opt 加载 minittir 方言"
	@echo "make dev-shell   - 起 linux/arm64 容器（内含 triton，用 TRITON_INTERPRET=1 跑 kernel 正确性）"
	@echo "make bw          - 实测 HBM 带宽（需 GPU；输出 benchmarks/results/<gpu>/bw_probe.json）"
	@echo "make gpu-verify  - GPU 侧一键验证（测试 + 带宽 + 冒烟）"

doctor:
	@echo "PY = $(PY)"
	@which -a python3 | head -5
	@$(PY) -c "import sys; print('exec:', sys.executable)" || true
	@$(PY) -c "import torch; print('torch:', torch.__version__)" 2>/dev/null || echo "torch: MISSING in $(PY)  ->  make test PY=/path/to/python3"

test:
	$(PY) tests/run_local.py

smoke:
	$(PY) -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__, '| cuda', torch.cuda.is_available())"
	$(PY) tests/run_local.py

dev-shell:
	docker build -f docker/Dockerfile.cpu -t tritonlab-cpu .
	docker run --rm -it -v "$$PWD":/w -w /w tritonlab-cpu

# 本机同时有 llc 和 ptxas 时直接跑（GPU 机器/容器内）；否则借容器（Mac：brew 的 llc 有 NVPTX，但没有 ptxas）
smoke-cubin:
	@if command -v llc >/dev/null 2>&1 && command -v ptxas >/dev/null 2>&1; then \
	  $(PY) -m minitriton.runtime.toolchain tests/fixtures/hello.ll --arch $(ARCH); \
	elif docker info >/dev/null 2>&1; then \
	  echo "[mac] 本机缺 ptxas，改用容器执行整条链（arch=$(ARCH)）"; \
	  docker run --rm -v "$$PWD":/w -w /w tritonlab-cpu \
	    python -m minitriton.runtime.toolchain tests/fixtures/hello.ll --arch $(ARCH); \
	else \
	  $(PY) -m minitriton.runtime.toolchain tests/fixtures/hello.ll --arch $(ARCH); \
	fi

mlir-spike:
	$(MAKE) -C mlir spike

bw:
	$(PY) bench/bw_probe.py

gpu-verify:
	bash scripts/gpu_verify.sh

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache dump
