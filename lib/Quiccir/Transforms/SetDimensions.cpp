//====- SetDimensions - Lowering Quiccir.View -----------------------------===//
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
struct QuiccirSetDimensions : public QuiccirSetDimensionsBase<QuiccirSetDimensions> {
  using QuiccirSetDimensionsBase<QuiccirSetDimensions>::QuiccirSetDimensionsBase;

  QuiccirSetDimensions(llvm::ArrayRef<int64_t> phys, llvm::ArrayRef<int64_t> mods);

  LogicalResult initializeOptions(StringRef options) final;

  void runOnOperation() final;

  private:
  llvm::SmallVector<int64_t, 3> _phys;
  llvm::SmallVector<int64_t, 3> _mods;
};
} // namespace

QuiccirSetDimensions::QuiccirSetDimensions(llvm::ArrayRef<int64_t> phys, llvm::ArrayRef<int64_t> mods) {
  for (auto&& p : phys) {
    _phys.push_back(p);
  }
  for (auto&& m : mods) {
    _mods.push_back(m);
  }
}

LogicalResult QuiccirSetDimensions::initializeOptions(StringRef options) {
  if (failed(Pass::initializeOptions(options)))
    return failure();
  for (auto&& p : physDim) {
      _phys.push_back(p);
  }
  for (auto&& m : modsDim) {
      _mods.push_back(m);
  }
  return success();
}

void setMissingDimensions(Value val, llvm::ArrayRef<std::int64_t> dims) {
  Type valTy = val.getType();
  if (auto tensor = valTy.dyn_cast<RankedTensorType>()) {
    auto shape = tensor.getShape();
    llvm::SmallVector<int64_t, 3> newShape(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == ShapedType::kDynamic) {
            newShape[i] = dims[i];
        }
        else {
            newShape[i] = shape[i];
        }
    }
    auto eleTy = tensor.getElementType();
    auto encoding = tensor.getEncoding();
    auto plusDymsTy = get<RankedTensorType>(val.getContext(), newShape, eleTy, encoding);
    val.setType(plusDymsTy);
  }
}

void QuiccirSetDimensions::runOnOperation() {
  // auto module = getOperation();
  auto *ctx = &getContext();

  RewritePatternSet patterns(ctx);
  ConversionTarget target(*ctx);

  // Walk from root func
  /// \todo clean-up with op traits
  WalkResult result = getOperation()->walk([&](Operation* op) {
      if (auto frIntOp = dyn_cast<FrIOp>(op)) {
        // set dimensions if not set
        Value phys = frIntOp.getPhys();
        llvm::SmallVector<int64_t, 3> physDim{_phys[0], _phys[1], _phys[2]};
        setMissingDimensions(phys, physDim);
        llvm::SmallVector<int64_t, 3> modsDim{_phys[0], _mods[1], _phys[2]};
        Value mods = frIntOp.getMods();
        setMissingDimensions(mods, modsDim);
      }
      if (auto alIntOp = dyn_cast<AlIOp>(op)) {
        // set dimensions if not set
        Value phys = alIntOp.getPhys();
        llvm::SmallVector<int64_t, 3> physDim{_mods[1], _phys[2], _phys[0]};
        setMissingDimensions(phys, physDim);
        llvm::SmallVector<int64_t, 3> modsDim{_mods[1], _mods[2], _phys[0]};
        Value mods = alIntOp.getMods();
        setMissingDimensions(mods, modsDim);
      }
      if (auto jwIntOp = dyn_cast<JWIOp>(op)) {
        // set dimensions if not set
        Value phys = jwIntOp.getPhys();
        llvm::SmallVector<int64_t, 3> physDim{_mods[2], _phys[0], _mods[1]};
        setMissingDimensions(phys, physDim);
        llvm::SmallVector<int64_t, 3> modsDim{_mods[2], _mods[0], _mods[1]};
        Value mods = jwIntOp.getMods();
        setMissingDimensions(mods, modsDim);
      }
      if (auto frPrjOp = dyn_cast<FrPOp>(op)) {
        // set dimensions if not set
        Value phys = frPrjOp.getPhys();
        llvm::SmallVector<int64_t, 3> physDim{_phys[0], _phys[1], _phys[2]};
        setMissingDimensions(phys, physDim);
        llvm::SmallVector<int64_t, 3> modsDim{_phys[0], _mods[1], _phys[2]};
        Value mods = frPrjOp.getMods();
        setMissingDimensions(mods, modsDim);
      }
      if (auto alPrjOp = dyn_cast<AlPOp>(op)) {
        // set dimensions if not set
        Value phys = alPrjOp.getPhys();
        llvm::SmallVector<int64_t, 3> physDim{_mods[1], _phys[2], _phys[0]};
        setMissingDimensions(phys, physDim);
        llvm::SmallVector<int64_t, 3> modsDim{_mods[1], _mods[2], _phys[0]};
        Value mods = alPrjOp.getMods();
        setMissingDimensions(mods, modsDim);
      }
      if (auto jwPrjOp = dyn_cast<JWPOp>(op)) {
        // set dimensions if not set
        Value phys = jwPrjOp.getPhys();
        llvm::SmallVector<int64_t, 3> physDim{_mods[2], _phys[0], _mods[1]};
        setMissingDimensions(phys, physDim);
        llvm::SmallVector<int64_t, 3> modsDim{_mods[2], _mods[0], _mods[1]};
        Value mods = jwPrjOp.getMods();
        setMissingDimensions(mods, modsDim);
      }
      // if (failed(deallocateBuffers(op)))
      //   return WalkResult::interrupt();
      return WalkResult::advance();
    });

  if (result.wasInterrupted())
    signalPassFailure();


  // if (failed(applyFullConversion(module, target, std::move(patterns))))
  //     signalPassFailure();
}

std::unique_ptr<Pass> mlir::quiccir::createSetDimensionsPass(
  llvm::ArrayRef<int64_t> phys,
  llvm::ArrayRef<int64_t> mods) {
  return std::make_unique<QuiccirSetDimensions>(phys, mods);
}