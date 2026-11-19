// TtirDialect 的实现：构造函数来自 tblgen，initialize() 由我们写。
#include "minittir/TtirDialect.h"
#include "minittir/TtirOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace minittir;

#include "minittir/TtirDialect.cpp.inc"

void TtirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "minittir/TtirOps.cpp.inc"
      >();
}
