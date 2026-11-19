// 插件入口：让 stock `mlir-opt` 认识我们的方言（以及后续的 pass）。
//
// 这是 out-of-tree 方案的关键：我们不构建自己的 driver，而是把
// libminittir.dylib 通过 --load-dialect-plugin / --load-pass-plugin 注入宿主。
#include "minittir/TtirDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

extern "C" ::mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return ::mlir::DialectPluginLibraryInfo{
      /*apiVersion=*/MLIR_PLUGIN_API_VERSION,
      /*pluginName=*/"minittir",
      /*pluginVersion=*/"0.1.0",
      /*registerDialectRegistryCallbacks=*/
      [](::mlir::DialectRegistry *registry) {
        registry->insert<::minittir::TtirDialect>();
      }};
}
