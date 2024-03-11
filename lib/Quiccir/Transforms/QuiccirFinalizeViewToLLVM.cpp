//====- QuiccirFinalizeViewToLLVM - Lowering Quiccir.View ----------------===//
//
// This file implements the final lowering of Quiccir.View to llvm.struct, i.e.
// in func/call/return ops.
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
struct QuiccirFinalizeViewToLLVMPass : public QuiccirFinalizeViewToLLVMBase<QuiccirFinalizeViewToLLVMPass> {
  using QuiccirFinalizeViewToLLVMBase<QuiccirFinalizeViewToLLVMPass>::QuiccirFinalizeViewToLLVMBase;
  void runOnOperation() final {
    auto module = getOperation();
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);

    quiccir::ViewTypeToPtrOfStructConverter typeConverter;

    // Populate with rules and apply rewriting rules.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
        typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    // All dynamic rules below accept new function, call, return
    // provided that all quiccir view types have been fully rewritten.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return typeConverter.isLegal(op);
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
    });

    // in patterns for other dialects.
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
    target.addLegalDialect<LLVM::LLVMDialect>();

    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::quiccir::createFinalizeViewToLLVMPass() {
  return std::make_unique<QuiccirFinalizeViewToLLVMPass>();
}
