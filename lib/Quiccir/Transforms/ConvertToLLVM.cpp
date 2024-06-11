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
    // Set dimensions
    Type I32Type = rewriter.getI32Type();
    Value dim0 = rewriter.create<LLVM::ConstantOp>(loc, I32Type,
      retViewTy.cast<ViewType>().getShape()[0]);
    Value dim1 = rewriter.create<LLVM::ConstantOp>(loc, I32Type,
      retViewTy.cast<ViewType>().getShape()[1]);
    Value dim2 = rewriter.create<LLVM::ConstantOp>(loc, I32Type,
      retViewTy.cast<ViewType>().getShape()[2]);
    // Insert dimensions
    // Note, swap logical order to match QuICC
    SmallVector<int64_t, 2> pos0 = {0, 2};
    Value retStruct0 = rewriter.create<LLVM::InsertValueOp>(loc, retStruct, dim0, pos0);
    SmallVector<int64_t, 2> pos1 = {0, 0};
    Value retStruct1 = rewriter.create<LLVM::InsertValueOp>(loc, retStruct0, dim1, pos1);
    SmallVector<int64_t, 2> pos2 = {0, 1};
    Value retStruct2 = rewriter.create<LLVM::InsertValueOp>(loc, retStruct1, dim2, pos2);

    // Copy memref struct size/ptrs to view struct
    // Get ptr ptr/pos
    SmallVector<int64_t, 1> ptrPtrPosMem = {1};
    Value ptrPtrOpaque = rewriter.create<LLVM::ExtractValueOp>(loc, ptrStruct, ptrPtrPosMem);
    // ptr to ptr<i32>
    Type i32PtrTy = mlir::LLVM::LLVMPointerType::get(I32Type);
    Value ptrPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, i32PtrTy, ptrPtrOpaque);
    // Set ptr ptr
    SmallVector<int64_t, 1> ptrPtrPosView = {3};
    Value retStruct3 = rewriter.create<LLVM::InsertValueOp>(loc, retStruct2, ptrPtr, ptrPtrPosView);
    // Get size ptr
    SmallVector<int64_t, 2> ptrSizePosMem = {3, 0};
    Value ptrSize64 = rewriter.create<LLVM::ExtractValueOp>(loc, ptrStruct, ptrSizePosMem);
    // i64 -> i32
    Value ptrSize32 = rewriter.create<LLVM::TruncOp>(loc, I32Type, ptrSize64);
    // Set size ptr
    SmallVector<int64_t, 1> ptrSizePosView = {4};
    Value retStruct4 = rewriter.create<LLVM::InsertValueOp>(loc, retStruct3, ptrSize32, ptrSizePosView);

    // Get ptr idx/coo
    SmallVector<int64_t, 1> idxPtrPosMem = {1};
    Value idxPtrOpaque = rewriter.create<LLVM::ExtractValueOp>(loc, idxStruct, idxPtrPosMem);
    // ptr to ptr<i32>
    Value idxPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, i32PtrTy, idxPtrOpaque);
    // Set ptr idx
    SmallVector<int64_t, 1> idxPtrPosView = {3};
    Value retStruct5 = rewriter.create<LLVM::InsertValueOp>(loc, retStruct4, idxPtr, idxPtrPosView);
    // Get size idx
    SmallVector<int64_t, 2> idxSizePosMem = {3, 0};
    Value idxSize64 = rewriter.create<LLVM::ExtractValueOp>(loc, idxStruct, idxSizePosMem);
    // i64 -> i32
    Value idxSize32 = rewriter.create<LLVM::TruncOp>(loc, I32Type, idxSize64);
    // Set size idx
    SmallVector<int64_t, 1> idxSizePosView = {4};
    Value retStruct6 = rewriter.create<LLVM::InsertValueOp>(loc, retStruct5, idxSize32, idxSizePosView);

    /// \todo data
    // Get ptr data
    SmallVector<int64_t, 1> dataPtrPosMem = {1};
    Value dataPtrOpaque = rewriter.create<LLVM::ExtractValueOp>(loc, dataStruct, dataPtrPosMem);
    // ptr to ptr<...32>
    Type eleTyMLIR = retViewTy.cast<ViewType>().getElementType();
    Type eleTyLLVM = llvmConverter.convertType(eleTyMLIR);
    Type eleTyptrTy = mlir::LLVM::LLVMPointerType::get(eleTyLLVM);
    Value dataPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, eleTyptrTy, dataPtrOpaque);
    // Set ptr data
    SmallVector<int64_t, 1> dataPtrPosView = {5};
    Value retStruct7 = rewriter.create<LLVM::InsertValueOp>(loc, retStruct6, dataPtr, dataPtrPosView);
    // Get size data
    SmallVector<int64_t, 2> dataSizePosMem = {3, 0};
    Value dataSize64 = rewriter.create<LLVM::ExtractValueOp>(loc, dataStruct, dataSizePosMem);
    // i64 -> i32
    Value dataSize32 = rewriter.create<LLVM::TruncOp>(loc, I32Type, dataSize64);
    // Set size data
    SmallVector<int64_t, 1> dataSizePosView = {6};
    Value retStruct8 = rewriter.create<LLVM::InsertValueOp>(loc, retStruct7, dataSize32, dataSizePosView);

    // Allocate on stack view struct
    quiccir::ViewTypeToPtrOfStructConverter ptrToStructConverter;
    Type bufPtrToStructType = ptrToStructConverter.convertType(retViewTy);
    Type I64Type = rewriter.getI64Type();
    Value one = rewriter.create<LLVM::ConstantOp>(loc, I64Type,
      rewriter.getIndexAttr(1));
    Value bufPtrStruct = rewriter.create<LLVM::AllocaOp>(loc, bufPtrToStructType, one);
    rewriter.create<LLVM::StoreOp>(loc, retStruct8, bufPtrStruct);

    // Replace op with cast back to view
    SmallVector<Value, 1> castOperands = {bufPtrStruct};
    auto newOp = rewriter.create<UnrealizedConversionCastOp>(loc, retViewTy, castOperands);
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
  target.addIllegalDialect<quiccir::QuiccirDialect>();
  // // Also we need alloc / materialize to be legal
  target.addLegalOp<
    quiccir::AllocDataOp
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
