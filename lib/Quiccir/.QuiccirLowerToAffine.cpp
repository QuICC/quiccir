//====- QuiccirLowerToAffine.cpp - Lowering from Quiccir to Affine+Std --===//
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
// QuiccirToAffine RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

static Value materializeToTensor(OpBuilder &builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(inputs[0].getType().isa<BaseMemRefType>());
  return builder.create<bufferization::ToTensorOp>(loc, type, inputs[0]);
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

/// Convert Tensor Input to buffer
static Value convertValueTensorToBuffer(Value tensor, PatternRewriter &rewriter) {
    auto tensorType = tensor.getType().cast<TensorType>();
    // auto shape = tensorType.getShape();
    // add check for dynamic shape
    // bufferize
    auto memRefType = convertTensorToMemRef(tensorType);
    return rewriter.create<bufferization::ToMemrefOp>(tensor.getLoc(), memRefType, tensor);
}

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
    auto b0Buffer = convertValueTensorToBuffer(b0, rewriter);
    auto b0Shape = b0.getType().cast<TensorType>().getShape();

    Value umod = bwdAdaptor.getUmod();
    auto umodBuffer = convertValueTensorToBuffer(umod, rewriter);
    auto umodShape = umod.getType().cast<TensorType>().getShape();

    Value b1 = bwdAdaptor.getB1();
    auto b1Buffer = convertValueTensorToBuffer(b1, rewriter);
    auto b1Shape = b1.getType().cast<TensorType>().getShape();

    // Insert an allocation and deallocation for the result of this operation.
    auto retTensorType = (*op->result_type_begin()).cast<TensorType>();
    auto retMemRefType = convertTensorToMemRef(retTensorType);
    auto retAlloc = insertAllocAndDealloc(retMemRefType, loc, rewriter);

    // need to alloc temporary
    MemRefType tmp = MemRefType::get({umodShape[0], b0Shape[0], umodShape[2]},
      retTensorType.getElementType());
    auto tmpBuffer = insertAllocAndDealloc(tmp, loc, rewriter);

    // set tmp to zero
    {
      SmallVector<int64_t, 3> lowerBounds(3, /*Value=*/0);
      SmallVector<int64_t, 3> upperBounds{umodShape[0], b0Shape[0], umodShape[2]};
      SmallVector<int64_t, 3> steps(3, /*Value=*/1);
      buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // const
          auto elementType = retTensorType.getElementType();
          auto zero = builder.create<arith::ConstantOp>(loc, elementType,
              builder.getFloatAttr(elementType, 0.0));
          builder.create<AffineStoreOp>(loc, zero, tmpBuffer, ivs);
        }
      );
    }

    // set ret to zero
    {
      SmallVector<int64_t, 3> lowerBounds(3, /*Value=*/0);
      SmallVector<int64_t, 3> upperBounds{umodShape[0], b0Shape[0], b1Shape[0]};
      SmallVector<int64_t, 3> steps(3, /*Value=*/1);
      buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // const
          auto elementType = retTensorType.getElementType();
          auto zero = builder.create<arith::ConstantOp>(loc, elementType,
              builder.getFloatAttr(elementType, 0.0));
          builder.create<AffineStoreOp>(loc, zero, retAlloc, ivs);
        }
      );
    }

    // tmp = b0*umod
    {
      SmallVector<int64_t, 4> lowerBounds(4, /*Value=*/0);
      SmallVector<int64_t, 4> upperBounds{umodShape[0], b0Shape[0], umodShape[2], b0Shape[1]};
      SmallVector<int64_t, 4> steps(4, /*Value=*/1);
      buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // m - k
          SmallVector<Value, 2> b0Ivs{ivs[1], ivs[3]};
          auto b0Loaded = builder.create<AffineLoadOp>(
                         loc, b0Buffer, b0Ivs);
          // k - n
          SmallVector<Value, 3> umodIvs{ivs[0], ivs[3], ivs[2]};
          auto umodLoaded = builder.create<AffineLoadOp>(
                         loc, umodBuffer, umodIvs);
          // m - n
          SmallVector<Value, 3> tmpIvs{ivs[0], ivs[1], ivs[2]};
          auto tmpLoaded = builder.create<AffineLoadOp>(
                         loc, tmpBuffer, tmpIvs);

          auto mul = builder.create<arith::MulFOp>(loc, b0Loaded, umodLoaded);
          auto tmpToStore = builder.create<arith::AddFOp>(loc, tmpLoaded, mul);

          builder.create<AffineStoreOp>(loc, tmpToStore, tmpBuffer, tmpIvs);
        }
      );
    }


    // ret = tmp*b1^T
    {
      SmallVector<int64_t, 4> lowerBounds(4, /*Value=*/0);
      SmallVector<int64_t, 4> upperBounds{umodShape[0], b0Shape[0], b1Shape[0], b1Shape[1]};
      SmallVector<int64_t, 4> steps(4, /*Value=*/1);
      buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // m - k
          SmallVector<Value, 2> tmpIvs{ivs[0], ivs[1], ivs[3]};
          auto tmpLoaded = builder.create<AffineLoadOp>(
                         loc, tmpBuffer, tmpIvs);
          // k - n
          SmallVector<Value, 2> b1Ivs{ivs[2], ivs[3]}; // transpose
          auto b1Loaded = builder.create<AffineLoadOp>(
                         loc, b1Buffer, b1Ivs);
          // m - n
          SmallVector<Value, 2> retIvs{ivs[0], ivs[1], ivs[2]};
          auto retLoaded = builder.create<AffineLoadOp>(
                         loc, retAlloc, retIvs);

          auto mul = builder.create<arith::MulFOp>(loc, tmpLoaded, b1Loaded);
          auto retToStore = builder.create<arith::AddFOp>(loc, retLoaded, mul);

          builder.create<AffineStoreOp>(loc, retToStore, retAlloc, retIvs);
        }
      );
    }

    // return value memref to tensor
    auto retTen = materializeToTensor(rewriter, retTensorType, retAlloc, loc);

    rewriter.replaceOp(op, retTen);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// QuiccirToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the quiccir operations
namespace {
struct QuiccirToAffineLoweringPass
    : public QuiccirLowerToAffineBase<QuiccirToAffineLoweringPass> {
  void runOnOperation() final;
};
} // namespace

void QuiccirToAffineLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arithmetic`, `Func`, and `MemRef` dialects.
  target
      .addLegalDialect<AffineDialect, BuiltinDialect, arith::ArithDialect,
                       func::FuncDialect, memref::MemRefDialect,
                       bufferization::BufferizationDialect>();

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
std::unique_ptr<Pass> mlir::quiccir::createLowerToAffinePass() {
  return std::make_unique<QuiccirToAffineLoweringPass>();
}
