// TTIR 各 op 的 C++ 类声明。生成物在 build/inc/minittir/TtirOps.h.inc。
#pragma once

#include "minittir/TtirDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "minittir/TtirOps.h.inc"
