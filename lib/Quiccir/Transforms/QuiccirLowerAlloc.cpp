//====- QuiccirLowerAlloc.cpp - Lowering from Quiccir Alloc ---------------===//
//
// This file implements a partial lowering of Quiccir Alloc to library calls.
//
//===----------------------------------------------------------------------===//

#include "Quiccir/Transforms/QuiccirPassDetail.h"

#include "Quiccir/IR/QuiccirDialect.h"
#include "Quiccir/IR/QuiccirOps.h"
#include "Quiccir/Transforms/QuiccirPasses.h"
#include "Quiccir/Transforms/TypeConverter.h"
#include "Quiccir/Transforms/Utils.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::quiccir;

namespace {

//===----------------------------------------------------------------------===//
// QuiccirToStd RewritePatterns: Alloc
//===----------------------------------------------------------------------===//

/// \todo this would cleanup a bit by ineriting from OpConverionsPattern
struct AllocOpLowering : public ConversionPattern {
  AllocOpLowering(MLIRContext *ctx)
      : ConversionPattern(quiccir::AllocOp::getOperationName(), /*benefit=*/1 , ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();

    quiccir::ViewTypeToStructConverter structConverter;
    quiccir::ViewTypeToPtrOfStructConverter ptrToStructConverter;

    // Insert llvm.struct
    Type retViewType = (*op->result_type_begin()).cast<ViewType>();
    Type bufStructType = structConverter.convertType(retViewType);
    // undef
    Value bufStructScratch = rewriter.create<LLVM::UndefOp>(loc, bufStructType);
    // set dims
    Type I32Type = rewriter.getI32Type();
    Value dim0 = rewriter.create<LLVM::ConstantOp>(loc, I32Type,
      retViewType.cast<ViewType>().getShape()[0]);
    Value dim1 = rewriter.create<LLVM::ConstantOp>(loc, I32Type,
      retViewType.cast<ViewType>().getShape()[1]);
    Value dim2 = rewriter.create<LLVM::ConstantOp>(loc, I32Type,
      retViewType.cast<ViewType>().getShape()[2]);
    // insert dims
    // swap logical order!
    SmallVector<int64_t, 2> pos0 = {0, 2};
    Value buf0 = rewriter.create<LLVM::InsertValueOp>(loc, bufStructScratch, dim0, pos0);
    SmallVector<int64_t, 2> pos1 = {0, 0};
    Value buf1 = rewriter.create<LLVM::InsertValueOp>(loc, buf0, dim1, pos1);
    SmallVector<int64_t, 2> pos2 = {0, 1};
    Value buf2 = rewriter.create<LLVM::InsertValueOp>(loc, buf1, dim2, pos2);

    // allocate on stack
    Type I64Type = rewriter.getI64Type();
    Value one = rewriter.create<LLVM::ConstantOp>(loc, I64Type,
      rewriter.getIndexAttr(1));
    Type bufPtrToStructType = ptrToStructConverter.convertType(retViewType);
    Value bufPtrStruct = rewriter.create<LLVM::AllocaOp>(loc, bufPtrToStructType, one);
    rewriter.create<LLVM::StoreOp>(loc, buf2, bufPtrStruct);

    // Replace op with cast
    SmallVector<Value, 1> castOperands = {bufPtrStruct};
    auto newOp = rewriter.create<UnrealizedConversionCastOp>(loc, retViewType, castOperands);
    rewriter.replaceOp(op, newOp);

    // Insert library call for alloc

    // Operands
    typename AllocOp::Adaptor adaptor(operands);
    Value viewProducer = adaptor.getProducerView();

    SmallVector <Type, 2> typeOperands = {retViewType, viewProducer.getType()};

    // return val becomes first operand
    auto libraryCallSymbol = getLibraryCallSymbolRef<AllocOp>(op, rewriter, typeOperands);
    if (failed(libraryCallSymbol))
      return failure();

    Value bufView = *op->result_begin();
    SmallVector<Value, 2> newOperands = {bufView, viewProducer};
    rewriter.create<func::CallOp>(
        loc, libraryCallSymbol->getValue(), TypeRange(), newOperands);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// QuiccirToStd RewritePatterns: AllocData
//===----------------------------------------------------------------------===//

struct AllocDataOpLowering : public ConversionPattern {
  AllocDataOpLowering(MLIRContext *ctx)
      : ConversionPattern(quiccir::AllocDataOp::getOperationName(), /*benefit=*/1 , ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();

    LLVMTypeConverter llvmConverter(getContext());
    // quiccir::ViewTypeToStructConverter structConverter;
    // quiccir::ViewTypeToPtrOfStructConverter ptrToStructConverter;

    // // Insert llvm.struct
    // Type retViewType = (*op->result_type_begin()).cast<ViewType>();
    // Type bufStructType = structConverter.convertType(retViewType);
    // // undef
    // Value bufStructScratch = rewriter.create<LLVM::UndefOp>(loc, bufStructType);
    // // set dims
    // Type I32Type = rewriter.getI32Type();
    // Value dim0 = rewriter.create<LLVM::ConstantOp>(loc, I32Type,
    //   retViewType.cast<ViewType>().getShape()[0]);
    // Value dim1 = rewriter.create<LLVM::ConstantOp>(loc, I32Type,
    //   retViewType.cast<ViewType>().getShape()[1]);
    // Value dim2 = rewriter.create<LLVM::ConstantOp>(loc, I32Type,
    //   retViewType.cast<ViewType>().getShape()[2]);
    // // insert dims
    // // swap logical order!
    // SmallVector<int64_t, 2> pos0 = {0, 2};
    // Value buf0 = rewriter.create<LLVM::InsertValueOp>(loc, bufStructScratch, dim0, pos0);
    // SmallVector<int64_t, 2> pos1 = {0, 0};
    // Value buf1 = rewriter.create<LLVM::InsertValueOp>(loc, buf0, dim1, pos1);
    // SmallVector<int64_t, 2> pos2 = {0, 1};
    // Value buf2 = rewriter.create<LLVM::InsertValueOp>(loc, buf1, dim2, pos2);

    // // allocate on stack
    // Type I64Type = rewriter.getI64Type();
    // Value one = rewriter.create<LLVM::ConstantOp>(loc, I64Type,
    //   rewriter.getIndexAttr(1));
    // Type bufPtrToStructType = ptrToStructConverter.convertType(retViewType);
    // Value bufPtrStruct = rewriter.create<LLVM::AllocaOp>(loc, bufPtrToStructType, one);
    // rewriter.create<LLVM::StoreOp>(loc, buf2, bufPtrStruct);

    // // Replace op with cast
    // SmallVector<Value, 1> castOperands = {bufPtrStruct};
    // auto newOp = rewriter.create<UnrealizedConversionCastOp>(loc, retViewType, castOperands);
    // rewriter.replaceOp(op, newOp);

    // // Insert library call for alloc

    // // Operands
    // typename AllocDataOp::Adaptor adaptor(operands);
    // Value viewProducer = adaptor.getProducerView();

    // SmallVector <Type, 2> typeOperands = {retViewType, viewProducer.getType()};

    // // return val becomes first operand
    // auto libraryCallSymbol = getLibraryCallSymbolRef<AllocDataOp>(op, rewriter, typeOperands);
    // if (failed(libraryCallSymbol))
    //   return failure();

    // Value bufView = *op->result_begin();
    // SmallVector<Value, 2> newOperands = {bufView, viewProducer};
    // rewriter.create<func::CallOp>(
    //     loc, libraryCallSymbol->getValue(), TypeRange(), newOperands);

    return success();
  }
};


//===----------------------------------------------------------------------===//
// QuiccirToStd RewritePatterns: Dealloc
//===----------------------------------------------------------------------===//

/// \todo this would cleanup a bit by ineriting from OpConverionsPattern
struct DeallocOpLowering : public ConversionPattern {
  DeallocOpLowering(MLIRContext *ctx)
      : ConversionPattern(quiccir::DeallocOp::getOperationName(), /*benefit=*/1 , ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();

    // Insert library call for dealloc

    // Operands
    typename DeallocOp::Adaptor adaptor(operands);
    Value view = adaptor.getView();

    // It might happen that we need to dealloc views with same layout
    // but different shape, so we insert a cast to a dynamic shape
    // to have the library call generic

    ViewType viewTy = view.getType().dyn_cast<ViewType>();
    SmallVector<int64_t, 3>  shape = {ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic};

    Type castViewTy = get<quiccir::ViewType>(view.getContext(), shape, viewTy.getElementType(), viewTy.getEncoding());
    auto castOp = rewriter.create<UnrealizedConversionCastOp>(loc, castViewTy, view);

    SmallVector <Type, 1> typeOperands = {castViewTy};

    // return val becomes first operand
    auto libraryCallSymbol = getLibraryCallSymbolRef<DeallocOp>(op, rewriter, typeOperands);
    if (failed(libraryCallSymbol))
      return failure();

    auto callOp = rewriter.create<func::CallOp>(
        loc, libraryCallSymbol->getValue(), TypeRange(), castOp.getResults());

    // Replace op with lib call
    rewriter.replaceOp(op, callOp);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// QuiccirAllocLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the quiccir operations
namespace {
struct QuiccirAllocLoweringPass
    : public QuiccirLowerAllocBase<QuiccirAllocLoweringPass> {
  void runOnOperation() final;
};
} // namespace

void QuiccirAllocLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
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
  // Also we need alloc / materialize to be legal
  // target.addLegalOp<quiccir::AllocOp, quiccir::MaterializeOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Quiccir operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<AllocOpLowering>(
      &getContext());
  patterns.add<AllocDataOpLowering>(
      &getContext());
  patterns.add<DeallocOpLowering>(
      &getContext());

  // void populateViewConversionPatterns(TypeConverter &typeConverter,
  // RewritePatternSet &patterns)
  // patterns.add<ViewTypeToPtrOfStructConverter>(converter, &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations to library calls
std::unique_ptr<Pass> mlir::quiccir::createLowerAllocPass() {
  return std::make_unique<QuiccirAllocLoweringPass>();
}
