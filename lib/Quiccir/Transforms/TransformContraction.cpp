//====- TransformContraction.cpp - Contract Transform ops ---------------===//
//
// This file implements a contraction of transform over addition and sum.
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

bool areSameTransform(Operation *lhsOp, Operation *rhsOp) {

  return false;
}


//===----------------------------------------------------------------------===//
// TransposeContration over AddOp or SubOp
//===----------------------------------------------------------------------===//
struct TransposeContractionOverAdd : public OpRewritePattern<AddOp> {
  TransposeContractionOverAdd(MLIRContext *ctx)
      : OpRewritePattern<AddOp>(ctx, /*benefit=*/1) {};

  LogicalResult
  matchAndRewrite(AddOp op,
                  PatternRewriter &rewriter) const final {
    // Get operands and check kinds
    mlir::Value addLhs = op.getLhs();
    mlir::Value addRhs = op.getRhs();

    // if not the same return failure
    // if (!transposeInputOp)
    //   return failure();

    // auto loc = op->getLoc();

    llvm::dbgs() << "done!\n";

    // Otherwise we can move the add upstream
    // rewriter.replaceOp(op, {transposeInputOp.getOperand()});

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// QuiccirTransformContractionPass
//===----------------------------------------------------------------------===//

/// This is a rewrite pass
namespace {
struct QuiccirTransformContractionPass
    : public QuiccirTransformContractionBase<QuiccirTransformContractionPass> {
    void runOnOperation() final;
};
} // namespace

void QuiccirTransformContractionPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());
  patterns.add<TransposeContractionOverAdd>(&getContext());

  FrozenRewritePatternSet patternSet(std::move(patterns));
  if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
    signalPassFailure();
}

/// Create a pass for lowering operations to library calls
std::unique_ptr<Pass> mlir::quiccir::createTransformContractionPass() {
  return std::make_unique<QuiccirTransformContractionPass>();
}
