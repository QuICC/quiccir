//====- ConvertToLLVM.cpp - Inserting Quiccir Dealloc -----------===//
//
// This file implements a pass that inserts DeallocOps.
// At the moment, for simplicity, we assume that the temporary buffers are not
// passed as arguments to other blocks (i.e. no control flow allowed).
// Otherwise a similar analysis as in BufferizationDeallocation would be needed.
//
//===----------------------------------------------------------------------===//

#include "Quiccir/Transforms/QuiccirPassDetail.h"

#include "Quiccir/IR/QuiccirDialect.h"
#include "Quiccir/IR/QuiccirOps.h"
#include "Quiccir/Transforms/QuiccirPasses.h"
#include "Quiccir/Transforms/TypeConverter.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::quiccir;

namespace {

//===----------------------------------------------------------------------===//
// QuiccirConvertToLLVMPass RewritePatterns: PointersOp
//===----------------------------------------------------------------------===//

/// \todo this would cleanup a bit by ineriting from OpConverionsPattern
struct PointerOpLowering : public ConversionPattern {
  PointerOpLowering(MLIRContext *ctx)
      : ConversionPattern(PointersOp::getOperationName(), /*benefit=*/1 , ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();

    // View Operand
    typename PointersOp::Adaptor adaptor(operands);
    Value operandView = *adaptor.getODSOperands(0).begin();

    // Struct Operand
    // Type converter
    quiccir::ViewTypeToStructConverter viewConverter;
    Type operandStructTy = viewConverter.convertType(operandView.getType());
    SmallVector<Value, 1> castOperandView = {operandView};
    auto operandStruct = rewriter.create<UnrealizedConversionCastOp>(loc, operandStructTy, castOperandView)->getResult(0);

    // ret memref needs replaced by a struct
    auto retMemTy = (*op->result_type_begin()).cast<MemRefType>();
    LLVMTypeConverter llvmConverter(getContext());

    auto retStructTy = llvmConverter.convertType(retMemTy);
    // alloca struct
    Value structPtr = rewriter.create<LLVM::UndefOp>(loc, retStructTy);

    // store on stack
    Type I64Type = rewriter.getI64Type();
    Value one = rewriter.create<LLVM::ConstantOp>(loc, I64Type,
      rewriter.getIndexAttr(1));
    Type ptrRetStructTy = mlir::LLVM::LLVMPointerType::get(retStructTy);
    Value bufRetStruct = rewriter.create<LLVM::AllocaOp>(loc, ptrRetStructTy, one);
    rewriter.create<LLVM::StoreOp>(loc, structPtr, bufRetStruct);


    // Set from ptr from view struct to memref struct
    SmallVector<int64_t, 1> ptrSizePosView = {2};
    Value sizePtr = rewriter.create<LLVM::ExtractValueOp>(loc, operandStruct, ptrSizePosView);
    // i32 -> i64
    Value extSizePtr = rewriter.create<LLVM::ZExtOp>(loc, I64Type, sizePtr);

    SmallVector<int64_t, 1> ptrSizePosMem = {3, 0};
    Value structPtr0 = rewriter.create<LLVM::InsertValueOp>(loc, structPtr,  extSizePtr, ptrSizePosMem);


    // cast back to memref
    SmallVector<Value, 1> castOperands = {structPtr0};
    auto newOp = rewriter.create<UnrealizedConversionCastOp>(loc, retMemTy, castOperands);
    rewriter.replaceOp(op, newOp);

    return success();
  }
};


} // namespace

//===----------------------------------------------------------------------===//
//  QuiccirConvertToLLVMPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the quiccir operations
namespace {
struct  QuiccirConvertToLLVMPass
    : public QuiccirConvertToLLVMBase<QuiccirConvertToLLVMPass> {
  void runOnOperation() final;
};
} // namespace

void  QuiccirConvertToLLVMPass::runOnOperation() {

  ConversionTarget target(getContext());

  // // Type converter
  // quiccir::ViewTypeToStructConverter viewConverter;

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target
      .addLegalDialect<BuiltinDialect, arith::ArithDialect,
                       func::FuncDialect, memref::MemRefDialect,
                       bufferization::BufferizationDialect,
                       LLVM::LLVMDialect>();

  // We also define the Quiccir dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted.
  // target.addIllegalDialect<quiccir::QuiccirDialect>();
  // // Also we need alloc / materialize to be legal
  target.addLegalOp<
    UnrealizedConversionCastOp
  //   quiccir::AllocOp,
  //   quiccir::MaterializeOp,
  //   quiccir::PointersOp,
  //   quiccir::IndicesOp,
  //   quiccir::AllocDataOp,
  //   quiccir::AssembleOp
    >();
  // target.addIllegalOp<quiccir::PointersOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Quiccir operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<PointerOpLowering>(
      &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations to library calls
std::unique_ptr<Pass> mlir::quiccir::createConvertToLLVMPass() {
  return std::make_unique<QuiccirConvertToLLVMPass>();
}
