//====- ViewWrapper.cpp - Inserting a wrapper using quiccir.view ----------===//
//
// This file implements a pass adding a wrapper quiccir.view -> tensor
// for the entry point function
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


struct QuiccirViewWrapperPass : public QuiccirViewWrapperBase<QuiccirViewWrapperPass> {
  using QuiccirViewWrapperBase<QuiccirViewWrapperPass>::QuiccirViewWrapperBase;
  void runOnOperation() final {
    auto module = getOperation();
    // auto *ctx = &getContext();

    // Count func ops, there should be only one at this point
    std::size_t nFunc = 0;
    module.walk([&](func::FuncOp funcOp) {
      ++nFunc;
    });

    if (nFunc > 1) {
      signalPassFailure();
      return;
    }

    module.walk([&](func::FuncOp funcOp) {
      OpBuilder builder(funcOp);

      // Set name
      std::string wrapperName = "wrapper_";
      wrapperName += funcOp.getSymName().str();

      // Set input / outputs types
      TensorToViewConverter cnv;
      auto funcTy = funcOp.getFunctionType();
      auto retsTy = funcTy.getResults();
      auto argsTy = funcTy.getInputs();
      auto nIn = funcTy.getNumInputs();
      SmallVector<Type, 4> viewArgsTy;
      for (auto ty : retsTy) {
        viewArgsTy.push_back(cnv.convertTensor(dyn_cast<RankedTensorType>(ty)));
      }
      for (auto ty : argsTy) {
        viewArgsTy.push_back(cnv.convertTensor(dyn_cast<RankedTensorType>(ty)));
      }

      // Insert func
      FunctionType viewFuncTy = FunctionType::get(
      builder.getContext(), viewArgsTy, {});
      builder.setInsertionPoint(funcOp);
      auto viewFuncOp = builder.create<func::FuncOp>(funcOp->getLoc(), wrapperName, viewFuncTy);
      auto loc = viewFuncOp->getLoc();
      // Within new func
      Block *viewFuncBody = viewFuncOp.addEntryBlock();
      builder.setInsertionPointToEnd(viewFuncBody);
      // Add casts view args -> tensors
      auto nArgs = viewFuncOp.getFunctionType().getNumInputs();
      SmallVector<Value, 4> viewValues;
      SmallVector<Value, 4> callValues;
      auto nOut = nArgs - nIn;
      for (unsigned i = 0; i < nIn; ++i) {
        // FuncOp has not operands, get them from block
        Value arg = viewFuncBody->getArguments()[nOut+i];
        // Value arg = viewFuncOp->getOperand(nOut+i);
        auto argCall = builder.create<UnrealizedConversionCastOp>(loc, argsTy[i], arg);
        callValues.push_back(argCall->getResult(0));
      }

      // Call to original tensor func
      auto call = builder.create<func::CallOp>(loc, funcOp, callValues);
      // Materialize returns to views

      builder.create<func::ReturnOp>(loc);

    });

  }
};
} // namespace

std::unique_ptr<Pass> mlir::quiccir::createViewWrapperPass() {
  return std::make_unique<QuiccirViewWrapperPass>();
}
