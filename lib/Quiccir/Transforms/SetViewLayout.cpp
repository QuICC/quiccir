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

  LogicalResult initializeOptions(StringRef options) final;

  void runOnOperation() final;

  private:
  SmallVector<std::string, 3> layInt{};
  SmallVector<std::string, 3> layPrj{};
};
} // namespace

LogicalResult QuiccirSetViewLayout::initializeOptions(StringRef options) {
  if (failed(Pass::initializeOptions(options)))
    return failure();
  layInt.push_back(layIntZero);
  layInt.push_back(layIntOne);
  layInt.push_back(layIntTwo);
  return success();
}

void setMissingLayout(Value val, llvm::StringRef enc) {
  Type valTy = val.getType();
  if (auto tensor = valTy.dyn_cast<RankedTensorType>()) {
    if (!tensor.getEncoding()) {
      Attribute encoding = get<StringAttr>(val.getContext(), enc);
      auto shape = tensor.getShape();
      auto eleTy = tensor.getElementType();
      auto plusAttrTy = get<RankedTensorType>(val.getContext(), shape, eleTy, encoding);
      val.setType(plusAttrTy);
      llvm::errs() << val << " ret has no ecoding\n";
    }
  }
}

void QuiccirSetViewLayout::runOnOperation() {
  auto module = getOperation();
  auto *ctx = &getContext();

  RewritePatternSet patterns(ctx);
  ConversionTarget target(*ctx);

  // Walk from root func
  WalkResult result = getOperation()->walk([&](Operation* op) {
      if (auto frIntOp = dyn_cast<FrIOp>(op)) {
        // check if attribute is set
        for (auto ret : op->getResults()) {
          setMissingLayout(ret, layInt[0]);
        }
      }
      if (auto frIntOp = dyn_cast<AlIOp>(op)) {
        // check if attribute is set
        for (auto ret : op->getResults()) {
          setMissingLayout(ret, layInt[1]);
        }
      }
      if (auto frIntOp = dyn_cast<JWIOp>(op)) {
        // check if attribute is set
        for (auto ret : op->getResults()) {
          setMissingLayout(ret, layInt[2]);
        }
      }

      // // return deallocateBuffers(op);
      // if (llvm::isa<QuiccirDialect>(op->getDialect())) {
      //   llvm::errs() << op->getName() << '\n';
      // }

      // if (failed(deallocateBuffers(op)))
      //   return WalkResult::interrupt();
      return WalkResult::advance();
    });

  if (result.wasInterrupted())
    signalPassFailure();


  // if (failed(applyFullConversion(module, target, std::move(patterns))))
  //     signalPassFailure();
}

std::unique_ptr<Pass> mlir::quiccir::createSetViewLayoutPass() {
  return std::make_unique<QuiccirSetViewLayout>();
}
