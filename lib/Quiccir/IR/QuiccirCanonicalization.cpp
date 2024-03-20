//===- QuiccirDialect.cpp - Quiccir dialect ---------------------*- C++ -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#include "Quiccir/IR/QuiccirDialect.h"
#include "Quiccir/IR/QuiccirOps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::quiccir;

/// Fold QuiccirOps with `tensor.cast` consumer if the `tensor.cast` has
/// result that is more static than the quiccir op.
struct FoldTensorCastConsumerOp : public OpRewritePattern<tensor::CastOp> {
  using OpRewritePattern<tensor::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    if (!tensor::canFoldIntoProducerOp(castOp))
      return failure();

    // need interface for quiccirOps
    // auto quiccirOp = castOp.getSource().getDefiningOp<QuiccirOp>();
    // if (!quiccirOp)
    //   return failure();

    // or check that op belongs to dialect
    auto quiccirOp = castOp.getSource().getDefiningOp();
    if (!llvm::isa<QuiccirDialect>(quiccirOp->getDialect()))
      return failure();

    // Cast can be in conditionally reachable region, if which case folding will
    // generate invalid code. Only conservatively fold ops in same block for
    // now.
    if (castOp->getBlock() != quiccirOp->getBlock())
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(quiccirOp);

    Location loc = quiccirOp->getLoc();
    OpResult resultValue = llvm::cast<OpResult>(castOp.getSource());
    unsigned resultNumber = resultValue.getResultNumber();
    auto resultType =
        llvm::cast<RankedTensorType>(castOp->getResult(0).getType());

    SmallVector<Type> resultTypes(quiccirOp->result_type_begin(),
                                  quiccirOp->result_type_end());
    resultTypes[resultNumber] = resultType;
    Operation *newOp = clone(rewriter, quiccirOp, resultTypes,
        quiccirOp->getOperands());

    // Create a tensor.cast operation back to the original type.
    Value castBack = rewriter.create<tensor::CastOp>(
        loc, resultValue.getType(), newOp->getResult(resultNumber));

    SmallVector<Value> results(newOp->result_begin(), newOp->result_end());
    results[resultNumber] = castBack;
    rewriter.replaceOp(quiccirOp, results);
    rewriter.replaceOp(castOp, newOp->getResult(resultNumber));
    return success();
  }
};

void QuiccirDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add<FoldTensorCastConsumerOp>(getContext());
}