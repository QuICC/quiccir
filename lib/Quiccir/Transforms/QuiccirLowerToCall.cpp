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
#include "llvm/Support/Error.h"
#include "llvm/Support/Errc.h"

using namespace mlir;
using namespace mlir::quiccir;

namespace {

//===----------------------------------------------------------------------===//
// QuiccirToStd RewritePatterns: Transform operations
//===----------------------------------------------------------------------===//

SmallVector<Value, 2> getIdxPtr(Operation* op, ConversionPatternRewriter &rewriter, Value operandBuffer)
{
  auto loc = op->getLoc();
  if (isa<TransposeOp>(op)) {
    // If the producer is a transpose Op, load meta from input

    // get function arguments
    auto func = op->getParentOp();
    // FuncOp has not operands, get them from block
    Block &funcBlock = func->getRegions()[0].getBlocks().front();
    std::uint32_t metaArrLoc = 0;
    Value ptrMetaArray = funcBlock.getArguments()[metaArrLoc];
    if (!ptrMetaArray.getType().isa<LLVM::LLVMPointerType>()) {
      func->emitError() << "expecting pointer as first func arg";
      return {};
    }

    // Load implementation array
    Value metaArray = rewriter.create<LLVM::LoadOp>(loc, ptrMetaArray);
    if (!metaArray.getType().isa<LLVM::LLVMArrayType>()) {
      func->emitError() << "expecting pointer to array";
      return {};
    }

    // Look at consumer to identify stage
    auto users = op->getUsers();
    if (users.empty()) {
      func->emitError() << "there is no user, cannot identify transform stage";
      return {};
    }
    Operation *user = *users.begin();
    SmallVector<int64_t, 1> indexPtr;
    SmallVector<int64_t, 1> indexIdx;
    if (isa<FrIOp>(user) || isa<FrPOp>(user)) {
      indexPtr.push_back(0);
      indexIdx.push_back(1);
    }
    if (isa<AlIOp>(user) || isa<AlPOp>(user)) {
      indexPtr.push_back(2);
      indexIdx.push_back(3);
    }
    else if (isa<JWIOp>(user) || isa<JWPOp>(user)) {
      indexPtr.push_back(4);
      indexIdx.push_back(5);
    }
    if (indexPtr.size() == 0 || indexIdx.size() == 0) {
      func->emitError() << "unable to recognize tranpose stage";
      return {};
    }
    Value ptrStruct = rewriter.create<LLVM::ExtractValueOp>(loc, metaArray, indexPtr);
    Type I32Type = rewriter.getI32Type();
    Type memTy = MemRefType::get({ShapedType::kDynamic}, I32Type);
    SmallVector<Value, 1> ptrOperands = {ptrStruct};
    Value ptr = rewriter.create<UnrealizedConversionCastOp>(loc, memTy, ptrOperands)->getResult(0);
    Value idxStruct = rewriter.create<LLVM::ExtractValueOp>(loc, metaArray, indexIdx);
    SmallVector<Value, 1> idxOperands = {idxStruct};
    Value idx = rewriter.create<UnrealizedConversionCastOp>(loc, memTy, idxOperands)->getResult(0);
    return {ptr, idx};
  }
  else {
    // Otherwise we get ptr and idx from producer view
    Type i32Type = IntegerType::get(op->getContext(), 32);
    Type metaTy =
    MemRefType::get({ShapedType::kDynamic}, i32Type);
    Value ptr = rewriter.create<PointersOp>(loc, metaTy, operandBuffer);
    Value idx = rewriter.create<IndicesOp>(loc, metaTy, operandBuffer);
    return {ptr, idx};
  }
}
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

    // Insert an allocation for each of the results of this operation.
    auto retTensorType = (*op->result_type_begin()).cast<TensorType>();

    // Here we should allocate a new view.
    // However, if the consumer is quicc.materialize we simply write in that buffer
    auto genRetBuffer = [&](Operation *op) -> llvm::Expected<SmallVector<Value, 3>> {
      SmallVector<Value, 3> buffers;
      for (auto indexedResult : llvm::enumerate(op->getResults())) {
        bool useExistingBuffer = false;
        Value result = indexedResult.value();
        // Check users
        for (Operation *user : result.getUsers()) {
          if (auto matOp = dyn_cast<MaterializeOp>(user)) {
            // One of the users is a materializeOp
            // no need to allocate buffer
            Value buffer = matOp.getView();
            rewriter.eraseOp(matOp);
            buffers.push_back(buffer);
            useExistingBuffer = true;
            break;
          }
        }
        if (!useExistingBuffer)
        {
          // Otherwise we need to allocate a new buffer
          auto ptrIdx = getIdxPtr(op, rewriter, operandBuffer);
          if (ptrIdx.size() < 2) {
            return llvm::createStringError(llvm::errc::invalid_argument,
              "could not retrieve meta data");
          }
          auto retTensorTy = result.getType().cast<TensorType>();
          ViewType retViewTy = getTypeConverter()->convertType(retTensorTy).cast<ViewType>();
          // Set lds for ops needing padding for FFT buffer
          if (isa<FrIOp>(op)) {
            auto operandTy = (operandBuffer.getType()).cast<ViewType>();
            int64_t lds = operandTy.getShape()[1]/2+1;
            if (lds > retViewTy.getShape()[1]) {
              retViewTy.setLds(lds);
            }
          }
          if (isa<TransposeOp>(op)) {
            // Check consumer, if FrPOp then the buffer might need padding
            auto users = op->getUsers();
            if (!users.empty()) {
              Operation *user = *users.begin();
              if (auto proj = dyn_cast<FrPOp>(user)) {
                auto physTy = proj.getPhys().getType().cast<RankedTensorType>();
                int64_t lds = physTy.getShape()[1]/2+1;
                if (lds > retViewTy.getShape()[1]) {
                  retViewTy.setLds(lds);
                }
              }
            }
          }
          Type I64Type = rewriter.getI64Type();
          int64_t lds = retViewTy.getShape()[1];
          // If lds is set, retrieve it
          if (retViewTy.getLds() != ShapedType::kDynamic) {
            lds = retViewTy.getLds();
          }
          Value ldsVal = rewriter.create<LLVM::ConstantOp>(loc, I64Type,
          rewriter.getI64IntegerAttr(lds));

          Type dataTy = MemRefType::get({ShapedType::kDynamic},
            retViewTy.getElementType());
          Value data = rewriter.create<AllocDataOp>(loc, dataTy, ptrIdx[0], ptrIdx[1], ldsVal,
            retViewTy.getEncoding().cast<StringAttr>().str());
          Value buffer = rewriter.create<AssembleOp>(loc, retViewTy, ptrIdx[0], ptrIdx[1], data);
          buffers.push_back(buffer);
        // Make sure to allocate at the beginning of the block.
        // auto *parentBlock = buffer.getDefiningOp()->getBlock();
        // buffer.getDefiningOp()->moveBefore(&parentBlock->front());
        }
      }
      // At this point there should be a buffer
      if (buffers.size() < 1) {
        return llvm::createStringError(llvm::errc::invalid_argument,
              "the buffer was not set correctly");
      }
      return buffers;
    };
    llvm::Expected<SmallVector<Value, 3>> ValueOrError = genRetBuffer(op);
    if (!ValueOrError) {
      op->emitError(toString(ValueOrError.takeError()));
      return failure();
    }
    SmallVector<Value, 3> retBuffers = ValueOrError.get();

    // get function arguments
    auto func = op->getParentOp();
    // FuncOp has not operands, get them from block
    Block &funcBlock = func->getRegions()[0].getBlocks().front();
    std::uint32_t thisArrLoc = 1;
    Value ptrImplArray = funcBlock.getArguments()[thisArrLoc];
    if (!ptrImplArray.getType().isa<LLVM::LLVMPointerType>()) {
      func->emitError() << "expecting pointer as second func arg";
      return failure();
    }

    // load implementation array
    Value implArray = rewriter.create<LLVM::LoadOp>(loc, ptrImplArray);
    if (!implArray.getType().isa<LLVM::LLVMArrayType>()) {
      func->emitError() << "expecting pointer to array.";
      return failure();
    }

    // get index from op attribute
    if (!cast<Top>(*op).getImplptr()) {
        return rewriter.notifyMatchFailure(op, "Implementation attribute is missing");
    }
    SmallVector<int64_t, 1> index = {static_cast<int64_t>(cast<Top>(*op).getImplptr().value())};
    auto implPtr = rewriter.create<LLVM::ExtractValueOp>(loc, implArray, index);

    // sub/add ops need extra cast to unknow dimensions
    // the same function might be called from different spaces
    // i.e. with different dimensions
    bool isAddSub = isa<SubOp>(op) || isa<AddOp>(op);

    if (isAddSub) {
      // cast operands and return vals to ?x?x?
      SmallVector<Value, 3> castNewOperOps;

      auto addCastOpers = [&](llvm::ArrayRef<Value> operands) {
        for (Value opers : operands) {
          auto oldTy = opers.getType().cast<ViewType>();
          SmallVector<int64_t, 3> shape(oldTy.getShape().size(), ShapedType::kDynamic);
          Type newViewTy = ViewType::get(shape, oldTy.getElementType(), oldTy.getEncoding());

          Value newOp = rewriter.create<UnrealizedConversionCastOp>(loc, newViewTy, opers)->getResult(0);
          castNewOperOps.push_back(newOp);
        }
      };

      addCastOpers(retBuffers);
      addCastOpers(operands);


      // lib call

      // opaque ptr to implementation becomes first operand
      SmallVector <Type, 4> typeOperands = {implPtr.getType()};

      auto addFunOpers = [&](llvm::ArrayRef<Value> operands) {
        for (auto val : operands) {
          typeOperands.push_back(val.getType());
        }
      };

      addFunOpers(castNewOperOps);

      // return val becomes second operand
      auto libraryCallSymbol = getLibraryCallSymbolRef<Top>(op, rewriter, typeOperands);
      if (failed(libraryCallSymbol))
        return failure();

      SmallVector<Value, 4> newOperands = {implPtr};
      for (auto ret : castNewOperOps) {
        newOperands.push_back(ret);
      }
      rewriter.create<func::CallOp>(
          loc, libraryCallSymbol->getValue(), TypeRange(), newOperands);

      // Replace old Op with casts
      SmallVector<Value, 3> castOps;
      for (auto ret : retBuffers) {
        Value newOp = rewriter.create<UnrealizedConversionCastOp>(loc, retTensorType, ret)->getResult(0);
        castOps.push_back(newOp);
      }
      assert(op->getNumResults() == castOps.size());
      rewriter.replaceOp(op, castOps);
    }
    else {
      // opaque ptr to implementation becomes first operand
      SmallVector <Type, 4> typeOperands = {implPtr.getType()};

      auto addFunOpers = [&](llvm::ArrayRef<Value> operands) {
        for (auto val : operands) {
          typeOperands.push_back(val.getType());
        }
      };

      addFunOpers(retBuffers);
      addFunOpers(operands);

      // return val becomes second operand
      auto libraryCallSymbol = getLibraryCallSymbolRef<Top>(op, rewriter, typeOperands);
      if (failed(libraryCallSymbol))
        return failure();

      SmallVector<Value, 4> newOperands = {implPtr};
      for (auto ret : retBuffers) {
        newOperands.push_back(ret);
      }
      for (auto val : operands) {
        newOperands.push_back(val);
      }
      rewriter.create<func::CallOp>(
          loc, libraryCallSymbol->getValue(), TypeRange(), newOperands);

      // Replace old Op with casts
      SmallVector<Value, 3> castOps;
      for (auto ret : retBuffers) {
        Value newOp = rewriter.create<UnrealizedConversionCastOp>(loc, retTensorType, ret)->getResult(0);
        castOps.push_back(newOp);
      }
      assert(op->getNumResults() == castOps.size());
      rewriter.replaceOp(op, castOps);
    }
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
  target.addLegalOp<
    quiccir::MaterializeOp,
    quiccir::PointersOp,
    quiccir::IndicesOp,
    quiccir::AllocDataOp,
    quiccir::AssembleOp
    >();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Quiccir operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<OpLowering<quiccir::JWPOp>>(
      &getContext(), viewConverter);
  patterns.add<OpLowering<quiccir::JWIOp>>(
      &getContext(), viewConverter);
  patterns.add<OpLowering<quiccir::AlPOp>>(
      &getContext(), viewConverter);
  patterns.add<OpLowering<quiccir::AlIOp>>(
      &getContext(), viewConverter);
  patterns.add<OpLowering<quiccir::FrPOp>>(
      &getContext(), viewConverter);
  patterns.add<OpLowering<quiccir::FrIOp>>(
      &getContext(), viewConverter);
  patterns.add<OpLowering<quiccir::TransposeOp>>(
      &getContext(), viewConverter);
  patterns.add<OpLowering<quiccir::AddOp>>(
      &getContext(), viewConverter);
  patterns.add<OpLowering<quiccir::SubOp>>(
      &getContext(), viewConverter);
  patterns.add<OpLowering<quiccir::MulConstOp>>(
      &getContext(), viewConverter);
  patterns.add<OpLowering<quiccir::CrossOp>>(
      &getContext(), viewConverter);
  patterns.add<OpLowering<quiccir::DotOp>>(
      &getContext(), viewConverter);

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
