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
  std::string _lay0;
};
} // namespace

LogicalResult QuiccirSetViewLayout::initializeOptions(StringRef options) {
  if (failed(Pass::initializeOptions(options)))
    return failure();
  _lay0 = setViewLayout;
  return success();
}

void QuiccirSetViewLayout::runOnOperation() {
  auto module = getOperation();
  auto *ctx = &getContext();

  RewritePatternSet patterns(ctx);
  ConversionTarget target(*ctx);

  llvm::errs() << _lay0 << '\n';

  // mapping needs to passed in as option
  // op -> layout, str to str? op to str?

  // walk the tree and process quiccir ops with missing
  // layout

  // Walk from root func
  WalkResult result = getOperation()->walk([&](Operation* op) {
      if (auto frIntOp = dyn_cast<FrIOp>(op)) {
        // check if attribute is set
        for (auto ret : op->getResults()) {
          Type retTy = ret.getType();
          if (auto tensor = retTy.dyn_cast<RankedTensorType>()) {
            if (!tensor.getEncoding()) {
              Attribute encoding = get<StringAttr>(ctx, _lay0);
              auto shape = tensor.getShape();
              auto eleTy = tensor.getElementType();
              auto plusAttrTy = get<RankedTensorType>(ctx, shape, eleTy, encoding);
              ret.setType(plusAttrTy);
              llvm::errs() << op->getName() << " ret has no ecoding\n";
            }
          }
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
