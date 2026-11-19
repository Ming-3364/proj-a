// TTIR 各 op 的 C++ 定义。一个 TU 里展开一次 GET_OP_CLASSES 即可。
#include "minittir/TtirOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace minittir;

#define GET_OP_CLASSES
#include "minittir/TtirOps.cpp.inc"
