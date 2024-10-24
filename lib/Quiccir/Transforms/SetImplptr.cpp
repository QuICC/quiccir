//====- SetImplptr.cpp - Set the unique implementation pointer ----------===//
//
// This file implements a pass setting a unique implementation pointer counter
// for each quiccir op
//
//===----------------------------------------------------------------------===//

#include "Quiccir/Transforms/QuiccirPassDetail.h"

#include "Quiccir/IR/QuiccirDialect.h"
#include "Quiccir/IR/QuiccirOps.h"
#include "Quiccir/Transforms/QuiccirPasses.h"
#include "Quiccir/Transforms/TypeConverter.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Errc.h"

using namespace mlir;
using namespace mlir::quiccir;

namespace mlir::quiccir
{
  #define GEN_PASS_DEF_QUICCIRSETIMPLPTR
  #include "Quiccir/Transforms/QuiccirPasses.h.inc"
} // namespace mlir::quiccir


namespace {

struct QuiccirSetImplptr : public quiccir::impl::QuiccirSetImplptrBase<QuiccirSetImplptr> {
  using QuiccirSetImplptrBase<QuiccirSetImplptr>::QuiccirSetImplptrBase;
  void runOnOperation() final {
    auto module = getOperation();
    auto *ctx = &getContext();

    // walk over all quiccir ops
    // keep track of counter and seen ops
    // ops are identified by a hash that includes
    //  - op name
    //  - operands types
    //  - optionally op kind attribute
    //  - optionally op perm attribute
    std::uint64_t counter = 0;
    std::map<llvm::hash_code, std::uint64_t> opMap;

    module.walk([&](Operation *op) {
      if (!llvm::isa<QuiccirDialect>(op->getDialect())) {
        return;
      }
      OpBuilder builder(op);
      /// \todo find you why *.int and *.prj hash to same value
      // auto hOp = hash_value(op->getName().getTypeID());
      auto hOp = hash_value(op->getName().getIdentifier());
      llvm::hash_code ht;
      // operands type hash
      llvm::hash_code hOpers = 0;
      for (Value op : op->getOperands()) {
        auto hTen = hash_value(op.getType().getTypeID());
        llvm::dbgs() << hTen << '\t';
        llvm::hash_code hLay = 0;
        llvm::hash_code hEleTy = 0;
        if (auto tensorTy = dyn_cast<RankedTensorType>(op.getType())) {
          hLay = hash_value(tensorTy.getEncoding());
          hEleTy = hash_value(tensorTy.getElementType().getTypeID());
        }
        hOpers = hash_combine(hOpers, hLay, hTen, hEleTy);
      }
      llvm::hash_code hRets = 0;
      for (Value op : op->getResults()) {
        auto hTen = hash_value(op.getType().getTypeID());
        llvm::hash_code hLay = 0;
        llvm::hash_code hEleTy = 0;
        if (auto tensorTy = dyn_cast<RankedTensorType>(op.getType())) {
          hLay = hash_value(tensorTy.getEncoding());
          hEleTy = hash_value(tensorTy.getElementType().getTypeID());
        }
        hOpers = hash_combine(hOpers, hLay, hTen, hEleTy);
      }

      if (auto kind = op->getAttr("kind")) {
        auto hKind = hash_value(kind);
        ht = hash_combine(hOp, hOpers, hRets, hKind);
      }
      else if (auto perm = op->getAttr("permutation")) {
        auto hPerm = hash_value(perm);
        ht = hash_combine(hOp, hOpers, hRets, hPerm);
      }
      else {
        ht = hash_combine(hOp, hOpers, hRets);
      }

      // llvm::dbgs() << op->getName()
      // << '\t' << op->hashProperties()
      // << '\t' << op->getName().getStringRef()
      // << '\t' << op->getName().getIdentifier()
      // << '\t' << hash_value(op->getName().getIdentifier())
      // << '\t' << hOp
      // << '\t' << hOpers
      // << '\n';

      // update map
      if(opMap.count(ht) == 0) {
        opMap[ht] = counter++;
      }
      Type I64Type = builder.getI64Type();
      mlir::IntegerAttr implptr = get<IntegerAttr>(ctx, I64Type, opMap[ht]);
      op->setAttr("implptr", implptr);
    });

  }
};
} // namespace

std::unique_ptr<Pass> mlir::quiccir::createSetImplptrPass() {
  return std::make_unique<QuiccirSetImplptr>();
}
