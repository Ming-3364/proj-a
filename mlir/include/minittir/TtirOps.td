// minitriton 的 TTIR 方言（tensor 级：无 layout、无线程）。
// 对照 Triton：include/triton/Dialect/Triton/IR/TritonOps.td
//
// 当前只有 1 个 op，用于验证「ODS → tblgen → 插件加载」这条链路。
// 后续按 docs/spec/language-subset.md 逐个补：load/store/arange/dot/reduce。
#ifndef MINITTR_TTIR_OPS
#define MINITTR_TTIR_OPS
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Ttir_Dialect : Dialect {
  let name = "ttir";
  let cppNamespace = "::minittir";
  let summary = "minitriton tensor-level IR (no layout, no threads)";
}

class Ttir_Op<string mnemonic, list<Trait> traits = []>
    : Op<Ttir_Dialect, mnemonic, traits>;

def Ttir_AddOp : Ttir_Op<"add", [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "elementwise tensor add";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
#endif // MINITTR_TTIR_OPS
