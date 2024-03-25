//====- SetViewLayout - Lowering Quiccir.View -----------------------------===//
//
// This file implements a pass to set the missing view layout attributes
//
//===----------------------------------------------------------------------===//

#include "Quiccir/Transforms/QuiccirPassDetail.h"

#include "Quiccir/IR/QuiccirDialect.h"
#include "Quiccir/IR/QuiccirOps.h"
#include "Quiccir/Transforms/QuiccirPasses.h"
#include "Quiccir/Transforms/TypeConverter.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

using namespace mlir;
using namespace mlir::quiccir;

namespace {
struct QuiccirSetViewLayout : public QuiccirSetViewLayoutBase<QuiccirSetViewLayout> {
  using QuiccirSetViewLayoutBase<QuiccirSetViewLayout>::QuiccirSetViewLayoutBase;
  void runOnOperation() final {
    auto module = getOperation();
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);



    // // in patterns for other dialects.
    // target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
    // target.addLegalDialect<LLVM::LLVMDialect>();

    // if (failed(applyFullConversion(module, target, std::move(patterns))))
    //     signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::quiccir::createSetViewLayoutPass() {
  return std::make_unique<QuiccirSetViewLayout>();
}
