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

bool isSameTransform(Operation *lhsOp, Operation *rhsOp) {
  /// \todo add transform/projection interface
  auto isTransform = [](Operation *op) {
    return isa<FrIOp>(op) || isa<FrPOp>(op)
      || isa<AlIOp>(op) || isa<AlPOp>(op)
      || isa<JWIOp>(op) || isa<JWPOp>(op);
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
// TransposeContration over linear operators: AddOp or SubOp
//===----------------------------------------------------------------------===//
template <class LINOP>
struct TransposeContractionOverLinOp : public OpRewritePattern<LINOP> {
  TransposeContractionOverLinOp<LINOP>(MLIRContext *ctx)
      : OpRewritePattern<LINOP>(ctx, /*benefit=*/1) {};

  LogicalResult
  matchAndRewrite(LINOP op,
                  PatternRewriter &rewriter) const final {
    // Get operands and check kinds
    Value addLhs = op.getLhs();
    Value addRhs = op.getRhs();

    // llvm::dbgs() << "lhs\t" << addLhs << '\n';

    auto prjOpLhs = addLhs.getDefiningOp();
    auto prjOpRhs = addRhs.getDefiningOp();
    // If there is no defining op, must be a func arg
    if (prjOpLhs == nullptr || prjOpRhs == nullptr) {
      return failure();
    }

    // If not the same transform return failure
    if (!isSameTransform(prjOpLhs, prjOpRhs)) {
      return failure();
    }

    // Otherwise we can move the add upstream the transpose

    // Transpose (projection) inputs
    Value prjLhs = prjOpLhs->getOperand(0);
    Value prjRhs = prjOpRhs->getOperand(0);
    // Transpose op ?
    auto traOpLhs = dyn_cast<TransposeOp>(prjLhs.getDefiningOp());
    auto traOpRhs = dyn_cast<TransposeOp>(prjRhs.getDefiningOp());

    if (traOpLhs && traOpRhs) {
      // get inputs
      Value traInLhs = traOpLhs.getInput();
      Value traInRhs = traOpRhs.getInput();

      // llvm::dbgs() << "contract!\n";

      auto loc = traOpLhs->getLoc();
      auto addNew = rewriter.create<LINOP>(loc, traInLhs, traInRhs);
      auto newTranspose = rewriter.clone(*static_cast<Operation*>(traOpLhs));
      newTranspose->setOperand(0, addNew);
      auto newProjector = rewriter.clone(*prjOpLhs);
      newProjector->setOperand(0, newTranspose->getResult(0));
      rewriter.replaceOp(op, newProjector);

      return success();
    }

    return failure();
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
  patterns.add<TransposeContractionOverLinOp<AddOp>>(&getContext());
  patterns.add<TransposeContractionOverLinOp<SubOp>>(&getContext());

  FrozenRewritePatternSet patternSet(std::move(patterns));
  if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
    signalPassFailure();
}

/// Create a pass for lowering operations to library calls
std::unique_ptr<Pass> mlir::quiccir::createTransformContractionPass() {
  return std::make_unique<QuiccirTransformContractionPass>();
}
