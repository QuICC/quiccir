//====- TransformGrouping.cpp - Group Transform ops ---------------===//
//
// This file implements a Grouping of N akin transform.
//
//===----------------------------------------------------------------------===//

#include "Quiccir/Transforms/QuiccirPassDetail.h"
#include "Quiccir/Transforms/QuiccirPasses.h"

#include "Quiccir/IR/QuiccirDialect.h"
#include "Quiccir/IR/QuiccirOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::quiccir;

namespace {

bool isSameTransform(Operation *lhsOp, Operation *rhsOp) {
  /// \todo add transform/projection interface
  auto isTransform = [](Operation *op) {
    return isa<FrIOp>(op) || isa<FrPOp>(op) || isa<AlIOp>(op) ||
           isa<AlPOp>(op) || isa<JWIOp>(op) || isa<JWPOp>(op);
  };
  auto isLhsTransform = isTransform(lhsOp);
  auto isRhsTransform = isTransform(lhsOp);
  if (isLhsTransform && isRhsTransform) {
    bool isSame = (lhsOp->getName() == rhsOp->getName());
    bool isSameKind = (lhsOp->getAttr("kind") == rhsOp->getAttr("kind"));
    return isSame && isSameKind;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// TransposeGrouping
//===----------------------------------------------------------------------===//
struct TransposeGrouping : public OpRewritePattern<TransposeOp> {
  TransposeGrouping(MLIRContext *ctx)
      : OpRewritePattern<TransposeOp>(ctx, /*benefit=*/1){};

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const final {

      return success();
    }

    return failure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// QuiccirTransformGroupingPass
//===----------------------------------------------------------------------===//

/// This is a rewrite pass
namespace {
struct QuiccirTransformGroupingPass
    : public QuiccirTransformGroupingBase<QuiccirTransformGroupingPass> {
  void runOnOperation() final;
};
} // namespace

void QuiccirTransformGroupingPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());
  patterns.add<TransposeGrouping>(&getContext());

  FrozenRewritePatternSet patternSet(std::move(patterns));
  if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
    signalPassFailure();
}

/// Create a pass for lowering operations to library calls
std::unique_ptr<Pass> mlir::quiccir::createTransformGroupingPass() {
  return std::make_unique<QuiccirTransformGroupingPass>();
}
