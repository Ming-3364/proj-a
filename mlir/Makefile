# minitriton 的 MLIR 部分（C++ 方言 + pass），以插件形式注入 stock mlir-opt。
#
# 为什么不用 CMake：Homebrew 的 llvm@20 不带任何 .cmake 配置（0 个文件），
# find_package(MLIR) 用不了。我们只需要 tblgen + clang++ 两条命令。
#
# 关键：插件链接时用 -undefined dynamic_lookup，MLIR 符号由宿主 mlir-opt 提供。
LLVM   ?= /opt/homebrew/opt/llvm@20
TBLGEN := $(LLVM)/bin/mlir-tblgen
CXX    := $(LLVM)/bin/clang++
MLIROP := $(LLVM)/bin/mlir-opt

TDFLAGS  := -I$(LLVM)/include -Iinclude
# llvm-config 给出的 cxxflags 包含 -fno-exceptions 等，跟着 LLVM 走避免 ABI 不匹配
CXXFLAGS := $(shell $(LLVM)/bin/llvm-config --cxxflags) -fPIC -fvisibility-inlines-hidden -Wall -Wextra -Wno-unused-parameter
INC      := -I$(LLVM)/include -Iinclude -Ibuild/inc

GEN := build/inc/minittir
LIB := build/libminittir.dylib
SRCS := lib/TtirDialect.cpp lib/TtirOps.cpp lib/Plugin.cpp
OBJS := $(patsubst lib/%.cpp,build/%.o,$(SRCS))
TD := include/minittir/TtirOps.td

.PHONY: all spike clean

all: $(LIB)

# ---- TableGen：4 个产物 ----
$(GEN)/TtirDialect.h.inc: $(TD)
	@mkdir -p $(GEN)
	$(TBLGEN) -gen-dialect-decls $(TD) $(TDFLAGS) -o $@

$(GEN)/TtirDialect.cpp.inc: $(TD)
	@mkdir -p $(GEN)
	$(TBLGEN) -gen-dialect-defs $(TD) $(TDFLAGS) -o $@

$(GEN)/TtirOps.h.inc: $(TD)
	@mkdir -p $(GEN)
	$(TBLGEN) -gen-op-decls $(TD) $(TDFLAGS) -o $@

$(GEN)/TtirOps.cpp.inc: $(TD)
	@mkdir -p $(GEN)
	$(TBLGEN) -gen-op-defs $(TD) $(TDFLAGS) -o $@

# ---- 编译与链接 ----
build/%.o: lib/%.cpp | $(GEN)/TtirDialect.h.inc $(GEN)/TtirDialect.cpp.inc $(GEN)/TtirOps.h.inc $(GEN)/TtirOps.cpp.inc
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(LIB): $(OBJS)
	$(CXX) -shared -undefined dynamic_lookup -o $@ $(OBJS)

# ---- 验收：股票 mlir-opt 能否加载我们的方言并原样打印 ----
spike: $(LIB)
	$(MLIROP) --load-dialect-plugin=$(LIB) tests/spike.mlir

clean:
	rm -rf build
