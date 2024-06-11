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
// QuiccirConvertToLLVMPass RewritePatterns: PointersOp and IndicesOp
//===----------------------------------------------------------------------===//

template <class Top>
struct MetaOpLowering : public ConversionPattern {
  MetaOpLowering(MLIRContext *ctx)
      : ConversionPattern(Top::getOperationName(), /*benefit=*/1 , ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();

    // View Operand
    typename Top::Adaptor adaptor(operands);
    Value operandView = *adaptor.getODSOperands(0).begin();

    // Struct Operand
    // Type converter
    quiccir::ViewTypeToStructConverter viewConverter;
    Type operandStructTy = viewConverter.convertType(operandView.getType());
    SmallVector<Value, 1> castOperandView = {operandView};
    auto operandStruct = rewriter.create<UnrealizedConversionCastOp>(loc, operandStructTy, castOperandView)->getResult(0);

    // ret memref needs replaced by a struct
    auto retMemTy = *op->result_type_begin();
    LLVMTypeConverter llvmConverter(getContext());
    auto retStructTy = llvmConverter.convertType(retMemTy);
    // Init struct
    Value structPtr = rewriter.create<LLVM::UndefOp>(loc, retStructTy);

    // Set size from view struct to memref struct
    int64_t shift = 0;
    if constexpr (std::is_same_v<Top, IndicesOp>) {
      shift += 2;
    }
    SmallVector<int64_t, 1> ptrSizePosView = {2+shift};
    Value ptrSize = rewriter.create<LLVM::ExtractValueOp>(loc, operandStruct, ptrSizePosView);
    // i32 -> i64
    Type I64Type = rewriter.getI64Type();
    Value extPtrSize = rewriter.create<LLVM::ZExtOp>(loc, I64Type, ptrSize);
    // Set size
    SmallVector<int64_t, 2> ptrSizePosMem = {3, 0};
    Value structPtr0 = rewriter.create<LLVM::InsertValueOp>(loc, structPtr, extPtrSize, ptrSizePosMem);
    // Set stride
    Value one = rewriter.create<LLVM::ConstantOp>(loc, I64Type,
      rewriter.getIndexAttr(1));
    SmallVector<int64_t, 2> stridePosMem = {4, 0};
    Value structPtr1 = rewriter.create<LLVM::InsertValueOp>(loc, structPtr0, one, stridePosMem);

    // Set ptr from view struct to memref struct
    SmallVector<int64_t, 1> ptrPtrPosView = {1+shift};
    Value ptrPtr = rewriter.create<LLVM::ExtractValueOp>(loc, operandStruct, ptrPtrPosView);
    // ptr<i32> to ptr
    Type opaquePtrTy = mlir::LLVM::LLVMPointerType::get(op->getContext());
    Value opaquePtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, opaquePtrTy, ptrPtr);
    // Set ptr buf
    SmallVector<int64_t, 1> ptrPtrPosMem = {0};
    Value structPtr2 = rewriter.create<LLVM::InsertValueOp>(loc, structPtr1, opaquePtr, ptrPtrPosMem);
    // Set aligned ptr
    SmallVector<int64_t, 1> ptrPtrAlPosMem = {1};
    Value structPtr3 = rewriter.create<LLVM::InsertValueOp>(loc, structPtr2, opaquePtr, ptrPtrAlPosMem);
    // Set offset
    Value zero = rewriter.create<LLVM::ConstantOp>(loc, I64Type,
      rewriter.getIndexAttr(0));
    SmallVector<int64_t, 1> offsetPos = {2};
    Value structPtr4 = rewriter.create<LLVM::InsertValueOp>(loc, structPtr3, zero, offsetPos);

    // Store on stack
    Type ptrRetStructTy = mlir::LLVM::LLVMPointerType::get(retStructTy);
    Value stackRetStruct = rewriter.create<LLVM::AllocaOp>(loc, ptrRetStructTy, one);
    rewriter.create<LLVM::StoreOp>(loc, structPtr4, stackRetStruct);

    // Load and cast back to memref
    Value retStruct = rewriter.create<LLVM::LoadOp>(loc, stackRetStruct);
    SmallVector<Value, 1> castOperands = {retStruct};
    auto newOp = rewriter.create<UnrealizedConversionCastOp>(loc, retMemTy, castOperands);
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// QuiccirConvertToLLVMPass RewritePatterns: AssembleOp
//===----------------------------------------------------------------------===//

struct AssembleOpLowering : public ConversionPattern {
  AssembleOpLowering(MLIRContext *ctx)
      : ConversionPattern(AssembleOp::getOperationName(), /*benefit=*/1 , ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();

    // MemRef Operands
    typename AssembleOp::Adaptor adaptor(operands);
    Value ptrMemRef = *adaptor.getODSOperands(0).begin();
    Value idxMemRef = *adaptor.getODSOperands(1).begin();
    Value dataMemRef = *adaptor.getODSOperands(2).begin();

    // Struct Operands
    // ptr
    LLVMTypeConverter llvmConverter(getContext());
    Type ptrStructTy = llvmConverter.convertType(ptrMemRef.getType());
    SmallVector<Value, 1> ptrCastOperand = {ptrMemRef};
    Value ptrStruct = rewriter.create<UnrealizedConversionCastOp>(loc, ptrStructTy, ptrCastOperand)->getResult(0);
    // idx
    Type idxStructTy = llvmConverter.convertType(idxMemRef.getType());
    SmallVector<Value, 1> idxCastOperand = {idxMemRef};
    Value idxStruct = rewriter.create<UnrealizedConversionCastOp>(loc, idxStructTy, idxCastOperand)->getResult(0);
    // data
    Type dataStructTy = llvmConverter.convertType(dataMemRef.getType());
    SmallVector<Value, 1> dataCastOperand = {dataMemRef};
    Value dataStruct = rewriter.create<UnrealizedConversionCastOp>(loc, dataStructTy, dataCastOperand)->getResult(0);

    // Init view struct
    quiccir::ViewTypeToStructConverter viewConverter;
    Type retViewTy = *op->result_type_begin();
    Type retStructTy = viewConverter.convertType(retViewTy);
    Value retStruct = rewriter.create<LLVM::UndefOp>(loc, retStructTy);
    /// \todo Set size

    // Copy memref struct size/ptrs to view struct
    // idx
    // get ptr
    SmallVector<int64_t, 1> idxPtrPosMem = {1};
    Value idxPtrOpaque = rewriter.create<LLVM::ExtractValueOp>(loc, idxStruct, idxPtrPosMem);
    // ptr to ptr<i32>
    Type I32Type = rewriter.getI32Type();
    Type i32PtrTy = mlir::LLVM::LLVMPointerType::get(I32Type);
    Value idxPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, i32PtrTy, idxPtrOpaque);
    // set ptr
    SmallVector<int64_t, 1> idxPtrPosView = {1};
    Value retStruct0 = rewriter.create<LLVM::InsertValueOp>(loc, retStruct, idxPtr, idxPtrPosView);
    /// \todo get/set size

    /// \todo ptr

    /// \todo data


    // Alloc on stack view struct

    // Cast back to view
    SmallVector<Value, 1> retCastOperand = {retStruct0};
    Value retView = rewriter.create<UnrealizedConversionCastOp>(loc, retViewTy, retCastOperand)->getResult(0);

    rewriter.replaceOp(op, retView);

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
  // target.addLegalOp<
    // UnrealizedConversionCastOp
  //   quiccir::AllocOp,
  //   quiccir::MaterializeOp,
  //   quiccir::PointersOp,
  //   quiccir::IndicesOp,
  //   quiccir::AllocDataOp,
  //   quiccir::AssembleOp
    // >();
  // target.addIllegalOp<quiccir::PointersOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Quiccir operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<MetaOpLowering<PointersOp>>(
      &getContext());
  patterns.add<MetaOpLowering<IndicesOp>>(
      &getContext());
  patterns.add<AssembleOpLowering>(
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
