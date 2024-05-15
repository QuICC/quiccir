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
#include "llvm/Support/Error.h"
#include "llvm/Support/Errc.h"

using namespace mlir;
using namespace mlir::quiccir;

namespace {


llvm::Expected<Type> setDimensionsEncoding(MLIRContext* ctx, Type valTy,
  llvm::ArrayRef<std::int64_t> dims, std::string encoding) {
  if (auto tensor = valTy.dyn_cast<RankedTensorType>()) {
    auto eleTy = tensor.getElementType();
    Attribute attEnc = get<StringAttr>(ctx, encoding);
    auto plusDymsTy = get<RankedTensorType>(ctx, dims, eleTy,attEnc);
    return plusDymsTy;
  }
  return llvm::createStringError(llvm::errc::invalid_argument, "Not a tensor");
}


struct QuiccirViewWrapperPass : public QuiccirViewWrapperBase<QuiccirViewWrapperPass> {
  using QuiccirViewWrapperBase<QuiccirViewWrapperPass>::QuiccirViewWrapperBase;
  void runOnOperation() final {
    auto module = getOperation();
    auto *ctx = &getContext();

    if (argsDim.size() == 0) {
      module->emitError("missing dim-args option");
    }
    if (retsDim.size() == 0) {
      module->emitError("missing dim-rets option");
    }

    // Count func ops, there should be only one at this point
    std::size_t nFunc = 0;
    module.walk([&](func::FuncOp funcOp) {
      ++nFunc;
    });

    if (nFunc > 1) {
      module->emitError("there should be only one function for this pass.");
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
        llvm::Expected<Type> TypeOrError = setDimensionsEncoding(ctx, ty, {1,1,1}, layRets);
        if (!TypeOrError) {
          module->emitError(toString(TypeOrError.takeError()));
        }
        viewArgsTy.push_back(cnv.convertTensor(dyn_cast<RankedTensorType>(TypeOrError.get())));
      }
      for (auto ty : argsTy) {
        llvm::Expected<Type> TypeOrError = setDimensionsEncoding(ctx, ty, {2,2,2}, layArgs);
        if (!TypeOrError) {
          module->emitError(toString(TypeOrError.takeError()));
        }
        viewArgsTy.push_back(cnv.convertTensor(dyn_cast<RankedTensorType>(TypeOrError.get())));
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
      SmallVector<Value, 4> callValues;
      auto nOut = nArgs - nIn;
      for (unsigned i = 0; i < nIn; ++i) {
        // FuncOp has not operands, get them from block
        Value arg = viewFuncBody->getArguments()[nOut+i];
        auto argCall = builder.create<UnrealizedConversionCastOp>(loc, argsTy[i], arg);
        callValues.push_back(argCall->getResult(0));
      }
      // Call to original tensor func
      auto call = builder.create<func::CallOp>(loc, funcOp, callValues);
      // Materialize returns to views
      for (unsigned i = 0; i < nOut; ++i) {
        // FuncOp has not operands, get them from block
        Value view = viewFuncBody->getArguments()[i];
        Value tensor = call->getResult(i);
        builder.create<quiccir::MaterializeOp>(loc, tensor, view);
      }
      // Block terminator
      builder.create<func::ReturnOp>(loc);
    });

  }
};
} // namespace

std::unique_ptr<Pass> mlir::quiccir::createViewWrapperPass(
  const QuiccirViewWrapperOptions &options) {
  return std::make_unique<QuiccirViewWrapperPass>();
}
