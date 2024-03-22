//===- QuiccirDialect.cpp - Quiccir dialect ---------------------*- C++ -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#include "Quiccir/IR/QuiccirDialect.h"
#include "Quiccir/IR/QuiccirOps.h"
#include "Quiccir/Interfaces/FoldTensorCastIntoConsumerOpInterface.cpp.inc"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "canonicalizer"

using namespace mlir;
using namespace mlir::quiccir;

//===----------------------------------------------------------------------===//
// Common Canonicalizers and Folders.
//===----------------------------------------------------------------------===//


namespace mlir
{
namespace quiccir
{

/// Fold QuiccirOps with `tensor.cast` consumer if the `tensor.cast` has
/// result that is more static than the quiccir op.
///
/// Example:
/// ```mlir
///   %1 = tensor.cast %0 : tensor<8x16xf32> to tensor<?x?xf32>
///   %2 = consumer %1 ... : tensor<?x?xf32> ...
/// ```
///
/// folds into:
///
/// ```mlir
///   %2 = consumer %0 ... : tensor<8x16xf32> ...
/// ```
struct FoldTensorCastProducerPattern
    : public OpInterfaceRewritePattern<FoldTensorCastIntoConsumerOpInterface> {
  using OpInterfaceRewritePattern<
      FoldTensorCastIntoConsumerOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(FoldTensorCastIntoConsumerOpInterface op,
                                PatternRewriter &rewriter) const override {
    // If no operand comes from a tensor::CastOp and can be folded then fail.
    bool hasTensorCastOperand =
        llvm::any_of(op->getOpOperands(), [&](OpOperand &opOperand) {
          if (llvm::isa<BlockArgument>(opOperand.get()))
            return false;
          auto castOp = opOperand.get().getDefiningOp<tensor::CastOp>();
          return castOp && canFoldIntoConsumerOp(castOp);
        });
    if (!hasTensorCastOperand)
      return failure();

    SmallVector<Value, 4> newOperands;
    newOperands.reserve(op->getNumOperands());
    for (OpOperand &opOperand : op->getOpOperands()) {
      auto tensorCastOp = opOperand.get().getDefiningOp<tensor::CastOp>();
      bool fold = canFoldIntoConsumerOp(tensorCastOp);
      newOperands.push_back(fold ? tensorCastOp.getOperand() : opOperand.get());
    }

    // Clone op.
    Operation *newOp = clone(rewriter, op, op->getResultTypes(), newOperands);
    // Replace
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

/// Fold QuiccirOps with `tensor.cast` consumer if the `tensor.cast` has
/// result that is more static than the quiccir op.
///
/// Example:
/// ```mlir
///   %1 = producer %0 ... : tensor<?x?xf32> ...
///   %2 = tensor.cast %1 : tensor<?x?xf32> to tensor<8x16xf32>
/// ```
///
/// folds into:
///
/// ```mlir
///   %2 = producer %0 ... : tensor<8x16xf32> ...
/// ```
struct FoldTensorCastConsumerPattern : public OpRewritePattern<tensor::CastOp> {
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

///
/// Canonicalize operations that can infer results or operands shape
///
struct InferShapePattern
    : public OpInterfaceRewritePattern<ShapeInferenceOpInterface> {
  using OpInterfaceRewritePattern<
      ShapeInferenceOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(ShapeInferenceOpInterface op,
                                PatternRewriter &rewriter) const override {

    // Ask the operation to infer its output/input shapes.
    LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << op->getName() << '\n');
    op.inferShapes();
    return success();
  }
};

void QuiccirDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add<quiccir::FoldTensorCastProducerPattern,
              quiccir::FoldTensorCastConsumerPattern,
              quiccir::InferShapePattern>(getContext());
}

} // namespace quiccir
} // namespace mlir
