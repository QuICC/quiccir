//====- QuiccirLowerToLinalg.cpp - Lowering from Quiccir to Linalg --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Quiccir operations to a combination of
// affine loops, memref operations and standard operations. This lowering
// expects that all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "Quiccir/QuiccirPassDetail.h"

#include "mlir/IR/BuiltinDialect.h"
#include "Quiccir/QuiccirDialect.h"
#include "Quiccir/QuiccirOps.h"
#include "Quiccir/QuiccirPasses.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::quiccir;

//===----------------------------------------------------------------------===//
// QuiccirToLinalg RewritePatterns
//===----------------------------------------------------------------------===//


namespace {

//===----------------------------------------------------------------------===//
// QuiccirToAffine RewritePatterns: Bwd operations
//===----------------------------------------------------------------------===//

struct QuadratureOpLowering : public ConversionPattern {
  QuadratureOpLowering(MLIRContext *ctx)
      : ConversionPattern(quiccir::QuadratureOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // tensor types
    quiccir::QuadratureOpAdaptor bwdAdaptor(operands);
    Value b0 = bwdAdaptor.getB0();
    auto b0Shape = b0.getType().cast<TensorType>().getShape();

    Value umod = bwdAdaptor.getUmod();
    auto umodShape = umod.getType().cast<TensorType>().getShape();

    Value b1 = bwdAdaptor.getB1();
    auto b1Ty = b1.getType().cast<TensorType>();
    auto b1Shape = b1Ty.getShape();


    // need to create tmp
    SmallVector<Value> dynDims;
    SmallVector<int64_t, 3> shapeDims = {umodShape[0], b0Shape[0], umodShape[2]};

    auto retTensorType = (*op->result_type_begin()).cast<TensorType>();
    auto elementType = retTensorType.getElementType();

    auto zeroAttr = rewriter.getZeroAttr(elementType);
    Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    auto initTmpTensor = rewriter.create<tensor::EmptyOp>(
        loc, shapeDims, retTensorType.getElementType(), dynDims);

    // set tmp to zero
    Value zeroTensor = rewriter
                           .create<linalg::FillOp>(loc, ValueRange{zero},
                                                   ValueRange{initTmpTensor})
                           .result();

    // set affine expressions
    // (e, i, j, k) -> (i, k)
    SmallVector<AffineExpr, 2> dimExprs13;
    dimExprs13.push_back(mlir::getAffineDimExpr(1, op->getContext()));
    dimExprs13.push_back(mlir::getAffineDimExpr(3, op->getContext()));
    // (e, i, j, k) -> (e, k, j)
    SmallVector<AffineExpr, 3> dimExprs032;
    dimExprs032.push_back(mlir::getAffineDimExpr(0, op->getContext()));
    dimExprs032.push_back(mlir::getAffineDimExpr(3, op->getContext()));
    dimExprs032.push_back(mlir::getAffineDimExpr(2, op->getContext()));
    // (e, i, j, k) -> (e, i, j)
    SmallVector<AffineExpr, 3> dimExprs012;
    dimExprs012.push_back(mlir::getAffineDimExpr(0, op->getContext()));
    dimExprs012.push_back(mlir::getAffineDimExpr(1, op->getContext()));
    dimExprs012.push_back(mlir::getAffineDimExpr(2, op->getContext()));
    // set indexing map
    SmallVector<AffineMap, 3> indexingMaps;
    unsigned dimCount = 4;
    indexingMaps.push_back(AffineMap::get(dimCount, /*symbolCount=*/0, dimExprs13, op->getContext()));
    indexingMaps.push_back(AffineMap::get(dimCount, /*symbolCount=*/0, dimExprs032, op->getContext()));
    indexingMaps.push_back(AffineMap::get(dimCount, /*symbolCount=*/0, dimExprs012, op->getContext()));
    // set iterator types
    SmallVector<utils::IteratorType> iteratorTypes {
      utils::IteratorType::parallel,
      utils::IteratorType::parallel,
      utils::IteratorType::parallel,
      utils::IteratorType::reduction};

    // tmp = b0*umod
    Value tmp = rewriter.create<linalg::GenericOp>(
      loc,
      /*resultTensorTypes=*/initTmpTensor.getType(),
      /*inputs=*/ValueRange{b0, umod},
      /*outputs=*/ValueRange{zeroTensor},
      indexingMaps,
      iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value mulTmp = nestedBuilder.create<arith::MulFOp>(nestedLoc, elementType, blockArgs[0], blockArgs[1]);
        Value opResult = nestedBuilder.create<arith::AddFOp>(nestedLoc, elementType, blockArgs[2], mulTmp);
        nestedBuilder.create<linalg::YieldOp>(loc, opResult);
      }
    )->getResult(0);

    // b1^t
    SmallVector<int64_t, 2> b1tShape = {b1Shape[1], b1Shape[0]};
    Value b1t = rewriter.create<tensor::EmptyOp>(
        loc, b1tShape, retTensorType.getElementType(), dynDims);
    b1t = linalg::makeTransposeOp(rewriter, loc, b1, b1t, {1, 0}).getResult(0);

    auto initRetTensor = rewriter.create<tensor::EmptyOp>(
        loc, retTensorType.getShape(), retTensorType.getElementType(), dynDims);

    // set ret to zero
    Value zeroRetTensor = rewriter
                           .create<linalg::FillOp>(loc, ValueRange{zero},
                                                   ValueRange{initRetTensor})
                           .result();


    // set affine expressions
    // (e, i, j, k) -> (e, i, k)
    SmallVector<AffineExpr, 3> dimExprs013;
    dimExprs013.push_back(mlir::getAffineDimExpr(0, op->getContext()));
    dimExprs013.push_back(mlir::getAffineDimExpr(1, op->getContext()));
    dimExprs013.push_back(mlir::getAffineDimExpr(3, op->getContext()));
    // (e, i, j, k) -> (k, j)
    SmallVector<AffineExpr, 2> dimExprs32;
    dimExprs32.push_back(mlir::getAffineDimExpr(3, op->getContext()));
    dimExprs32.push_back(mlir::getAffineDimExpr(2, op->getContext()));
    // replace operand maps
    indexingMaps[0] = AffineMap::get(dimCount, /*symbolCount=*/0, dimExprs013, op->getContext());
    indexingMaps[1] = AffineMap::get(dimCount, /*symbolCount=*/0, dimExprs32, op->getContext());

    // ret = tmp*b1^t
    Value ret = rewriter.create<linalg::GenericOp>(
      loc,
      /*resultTensorTypes=*/initRetTensor.getType(),
      /*inputs=*/ValueRange{tmp, b1t},
      /*outputs=*/ValueRange{zeroRetTensor},
      indexingMaps,
      iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value mulTmp = nestedBuilder.create<arith::MulFOp>(nestedLoc, elementType, blockArgs[0], blockArgs[1]);
        Value opResult = nestedBuilder.create<arith::AddFOp>(nestedLoc, elementType, blockArgs[2], mulTmp);
        nestedBuilder.create<linalg::YieldOp>(loc, opResult);
      }
    )->getResult(0);

    // set ret to zero
    rewriter.replaceOp(op, ret);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// QuiccirToLinalgLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the quiccir operations
namespace {
struct QuiccirToLinalgLoweringPass
    : public QuiccirLowerToLinalgBase<QuiccirToLinalgLoweringPass> {
  void runOnOperation() final;
};
} // namespace

void QuiccirToLinalgLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arithmetic`, `Func`, and `MemRef` dialects.
  target
      .addLegalDialect<linalg::LinalgDialect,
                       tensor::TensorDialect,
                       arith::ArithDialect,
                       func::FuncDialect>();

  // We also define the Quiccir dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted.
  target.addIllegalDialect<quiccir::QuiccirDialect>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Quiccir operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<QuadratureOpLowering>(
      &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects
std::unique_ptr<Pass> mlir::quiccir::createLowerToLinalgPass() {
  return std::make_unique<QuiccirToLinalgLoweringPass>();
}
