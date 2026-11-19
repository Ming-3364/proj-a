# MLIR out-of-tree 环境配方（已实测通过）

> 目标：不构建 LLVM、不写 CMake，用 brew 的预编译 MLIR，把我们的方言以**插件**形式注入 `mlir-opt`。
> 状态：**spike 已通过** —— `mlir-opt --load-dialect-plugin=build/libminittir.dylib tests/spike.mlir` 能解析打印 `ttir.add`。

## 环境事实（本机实测）

| 项 | 值 |
|---|---|
| LLVM | Homebrew `llvm@20` **20.1.8**，路径 `/opt/homebrew/opt/llvm@20` |
| 有 | `mlir-tblgen`、`mlir-opt`、`mlir-translate`、`llc`、`include/mlir`（26 个目录）、329 个 `libMLIR*.dylib` |
| 没有 | **任何 `.cmake` 配置**（`find -name '*.cmake'` = 0），所以 `find_package(MLIR)` 不可用 |
| 没有 | Python bindings |

结论：**Makefile + tblgen + clang++**，不依赖 CMake。

## 四个 TableGen 产物

```bash
B=/opt/homebrew/opt/llvm@20
$B/bin/mlir-tblgen -gen-dialect-decls TtirOps.td -I $B/include -o inc/TtirDialect.h.inc
$B/bin/mlir-tblgen -gen-dialect-defs  TtirOps.td -I $B/include -o inc/TtirDialect.cpp.inc
$B/bin/mlir-tblgen -gen-op-decls      TtirOps.td -I $B/include -o inc/TtirOps.h.inc
$B/bin/mlir-tblgen -gen-op-defs       TtirOps.td -I $B/include -o inc/TtirOps.cpp.inc
```

生成的代码形状（决定了头文件怎么写）：

- `TtirDialect.h.inc`：自带 `namespace minittir { class TtirDialect ... }` + `MLIR_DECLARE_EXPLICIT_TYPE_ID`
- `TtirDialect.cpp.inc`：构造函数（内部调 `initialize()`）+ `MLIR_DEFINE_EXPLICIT_TYPE_ID`
- `TtirOps.h.inc`：先 `namespace minittir { class AddOp; }`，然后 `#ifdef GET_OP_CLASSES` 块
- `TtirOps.cpp.inc`：`#ifdef GET_OP_LIST`（供 `addOperations<>`）与 `GET_OP_CLASSES` 定义

所以头文件三层：

```cpp
// TtirDialect.h
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "minittir/TtirDialect.h.inc"

// TtirOps.h
#include "minittir/TtirDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"   // ★ 见坑 1
#define GET_OP_CLASSES
#include "minittir/TtirOps.h.inc"
```

`.cpp` 侧：

```cpp
// TtirDialect.cpp
#include "minittir/TtirDialect.cpp.inc"      // 构造函数
void TtirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "minittir/TtirOps.cpp.inc"
      >();
}

// TtirOps.cpp
#define GET_OP_CLASSES
#include "minittir/TtirOps.cpp.inc"
```

## 插件入口

```cpp
extern "C" ::mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "minittir", "0.1.0",
          [](::mlir::DialectRegistry *r) { r->insert<::minittir::TtirDialect>(); }};
}
```

链接（macOS）：

```bash
clang++ -shared -undefined dynamic_lookup -o build/libminittir.dylib build/*.o
```

`-undefined dynamic_lookup` 是关键：MLIR 符号留给宿主 `mlir-opt` 解析，插件不链接 329 个 dylib。

## 四个坑（都真实踩到了）

| # | 坑 | 现象 | 解法 |
|---|---|---|---|
| 1 | `Pure` 的依赖没带全 | `no member named 'ConditionallySpeculatable' in namespace 'mlir'` | `.td` 里 `include "mlir/Interfaces/SideEffectInterfaces.td"`；`.h` 里 `#include "mlir/Interfaces/SideEffectInterfaces.h"` |
| 2 | `SameOperandsAndResultType` 不是可用的 td trait | `Variable not defined: 'SameOperandsAndResultType'` | 用 `AllTypesMatch<["lhs","rhs","result"]>`（在 `OpBase.td`） |
| 3 | `assemblyFormat` 没给 result 类型 | `type of result #0 ... is not buildable` | 写 `:` `type($result)`（或用 inferType 类 trait） |
| 4 | macOS `sed -i` | `invalid command code T` | 用 `sed -i ''` |

## 验证

```bash
make -C mlir spike        # 或根目录 make mlir-spike
```

期望输出：

```mlir
module {
  func.func @add(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = ttir.add %arg0, %arg1 : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}
```

已纳入回归：`tests/test_mlir_plugin.py`（没有 mlir-opt 时自动 skip）。

## 下一步

1. 扩充 TTIR op 集合（`load/store/arange/const`），按 `docs/spec/language-subset.md`。
2. 加第一个 pass：`ttir-to-ttgir`（layout 与线程映射的诞生处）。
3. 加 `mlir-opt --load-pass-plugin` 的验证用例。
