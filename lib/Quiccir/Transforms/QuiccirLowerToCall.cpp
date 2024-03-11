//====- QuiccirLowerToCall.cpp - Lowering from Quiccir to func.call -------===//
//
// This file implements a partial lowering of Quiccir to library calls.
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::quiccir;

namespace {

//===----------------------------------------------------------------------===//
// QuiccirToStd RewritePatterns: Jones Worland operations
//===----------------------------------------------------------------------===//

/// \todo this would cleanup a bit by ineriting from OpConverionsPattern
template <class Top>
struct OpLowering : public ConversionPattern {
  OpLowering(MLIRContext *ctx, TypeConverter &typeConverter)
      : ConversionPattern(typeConverter, Top::getOperationName(), /*benefit=*/1 , ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();

    // Operands (converted)
    typename Top::Adaptor adaptor(operands);
    Value operandBuffer = *adaptor.getODSOperands(0).begin();

    // Insert an allocation for the result of this operation.
    auto retTensorType = (*op->result_type_begin()).cast<TensorType>();
    auto retViewType = getTypeConverter()->convertType(retTensorType);

    // Here we should allocate a new view.
    // However, if the consumer is quicc.materialize we simply write in that buffer
    auto genRetBuffer = [&](Operation *op) -> Value {
      for (auto indexedResult : llvm::enumerate(op->getResults())) {
        Value result = indexedResult.value();
        if (result.hasOneUse()) {
          /// single use, check if the user is materializeOP
          Operation *user = *result.user_begin();
          if (auto op = dyn_cast<MaterializeOp>(user)) {
            Value buffer = op.getView();
            rewriter.eraseOp(op);
            return buffer;
          }
        }
      }
      // otherwise we need to insert a quicc.alloc
      Value buffer = rewriter.create<AllocOp>(loc, retViewType, operandBuffer, op->getName().getStringRef());
      // Make sure to allocate at the beginning of the block.
      // auto *parentBlock = buffer.getDefiningOp()->getBlock();
      // buffer.getDefiningOp()->moveBefore(&parentBlock->front());
      return buffer;
    };
    Value retBuffer = genRetBuffer(op);

    // get function arguments
    auto func = op->getParentOp();
    // FuncOp has not operands, get them from block
    Block &funcBlock = func->getRegions()[0].getBlocks().front();
    Value ptrImplArray = funcBlock.getArguments()[0];
    if (!ptrImplArray.getType().isa<LLVM::LLVMPointerType>()) {
      func->emitError() << "expecting pointer as first func arg.";
      return failure();
    }

    // load implementation array
    Value implArray = rewriter.create<LLVM::LoadOp>(loc, ptrImplArray);
    if (!implArray.getType().isa<LLVM::LLVMArrayType>()) {
      func->emitError() << "expecting pointer to array.";
      return failure();
    }

    // get index from op attribute
    ArrayRef<int64_t> index = cast<Top>(*op).getImplptr().value();
    auto implPtr = rewriter.create<LLVM::ExtractValueOp>(loc, implArray, index);

    // opaque ptr to implementation becomes first operand
    SmallVector <Type, 4> typeOperands = {implPtr.getType(), retViewType, operandBuffer.getType()};

    // return val becomes second operand
    auto libraryCallSymbol = getLibraryCallSymbolRef<Top>(op, rewriter, typeOperands);
    if (failed(libraryCallSymbol))
      return failure();

    SmallVector<Value, 4> newOperands = {implPtr, retBuffer, operandBuffer};
    rewriter.create<func::CallOp>(
        loc, libraryCallSymbol->getValue(), TypeRange(), newOperands);

    SmallVector<Value, 1> castOperands = {retBuffer};
    auto newOp = rewriter.create<UnrealizedConversionCastOp>(loc, retTensorType, castOperands);
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// QuiccirToCallLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the quiccir operations
namespace {
struct QuiccirToCallLoweringPass
    : public QuiccirLowerToCallBase<QuiccirToCallLoweringPass> {
  void runOnOperation() final;
};
} // namespace

void QuiccirToCallLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // Type converter
  quiccir::TensorToViewConverter viewConverter;

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
  target.addLegalOp<quiccir::AllocOp, quiccir::MaterializeOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Quiccir operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<OpLowering<quiccir::JWPOp>>(
      &getContext(), viewConverter);
   patterns.add<OpLowering<quiccir::JWIOp>>(
      &getContext(), viewConverter);

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
std::unique_ptr<Pass> mlir::quiccir::createLowerToCallPass() {
  return std::make_unique<QuiccirToCallLoweringPass>();
}
