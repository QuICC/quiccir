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

  QuiccirSetViewLayout(const std::array<std::array<std::string, 2>, 3> &layout) :
  layout(layout) {};

  LogicalResult initializeOptions(StringRef options) final;

  void runOnOperation() final;

  private:
  std::array<std::array<std::string, 2>, 3> layout;
};
} // namespace

LogicalResult QuiccirSetViewLayout::initializeOptions(StringRef options) {
  if (failed(Pass::initializeOptions(options)))
    return failure();
  if (layZero.size() == 2) {
    layout[0][0] = layZero[0];
    layout[0][1] = layZero[1];
  }
  else {
    layout[0][0] = "layPPP";
    layout[0][1] = "layMPP";
  }
  if (layOne.size() == 2) {
    layout[1][0] = layOne[0];
    layout[1][1] = layOne[1];
  }
  else {
    layout[1][0] = "layPMP";
    layout[1][1] = "layMMP";
  }
   if (layTwo.size() == 2) {
    layout[2][0] = layTwo[0];
    layout[2][1] = layTwo[1];
  }
  else {
    layout[2][0] = "layPMM";
    layout[2][1] = "layMMM";
  }
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
    }
  }
}

void QuiccirSetViewLayout::runOnOperation() {
  auto *ctx = &getContext();

  RewritePatternSet patterns(ctx);
  ConversionTarget target(*ctx);

  // Walk from root func
  /// \todo clean-up with op traits
  WalkResult result = getOperation()->walk([&](Operation* op) {
      if (auto frIntOp = dyn_cast<FrIOp>(op)) {
        // set attributes if not set
        Value phys = frIntOp.getPhys();
        setMissingLayout(phys, layout[0][0]);
        Value mods = frIntOp.getMods();
        setMissingLayout(mods, layout[0][1]);
       }
      if (auto frPrjOp = dyn_cast<FrPOp>(op)) {
        // set attributes if not set
        Value phys = frPrjOp.getPhys();
        setMissingLayout(phys, layout[0][0]);
        Value mods = frPrjOp.getMods();
        setMissingLayout(mods, layout[0][1]);
       }
      if (auto alIntOp = dyn_cast<AlIOp>(op)) {
        // set attributes if not set
        Value phys = alIntOp.getPhys();
        setMissingLayout(phys, layout[1][0]);
        Value mods = alIntOp.getMods();
        setMissingLayout(mods, layout[1][1]);
      }
      if (auto alPrjOp = dyn_cast<AlPOp>(op)) {
        // set attributes if not set
        Value phys = alPrjOp.getPhys();
        setMissingLayout(phys, layout[1][0]);
        Value mods = alPrjOp.getMods();
        setMissingLayout(mods, layout[1][1]);
      }
      if (auto jwIntOp = dyn_cast<JWIOp>(op)) {
        // set attributes if not set
        Value phys = jwIntOp.getPhys();
        setMissingLayout(phys, layout[2][0]);
        Value mods = jwIntOp.getMods();
        setMissingLayout(mods, layout[2][1]);
      }
      if (auto jwPrjOp = dyn_cast<JWPOp>(op)) {
        // set attributes if not set
        Value phys = jwPrjOp.getPhys();
        setMissingLayout(phys, layout[2][0]);
        Value mods = jwPrjOp.getMods();
        setMissingLayout(mods, layout[2][1]);
      }
      return WalkResult::advance();
    });

  if (result.wasInterrupted())
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::quiccir::createSetViewLayoutPass(
  const std::array<std::array<std::string, 2>, 3> &layout) {
  return std::make_unique<QuiccirSetViewLayout>(layout);
}