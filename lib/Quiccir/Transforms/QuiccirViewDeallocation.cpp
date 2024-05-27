//====- QuiccirViewDeallocation.cpp - Inserting Quiccir Dealloc -----------===//
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

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::quiccir;

namespace {

//===----------------------------------------------------------------------===//
// QuiccirViewDeallocation
//===----------------------------------------------------------------------===//
Operation *getEndOperation(Value value, Operation *startOperation) {

  Block *block = startOperation->getBlock();
  // Resolve the last operation (must exist by definition).
  Operation *endOperation = startOperation;
  for (Operation *useOp : value.getUsers()) {
    // Find the associated operation in the current block (if any).
    useOp = block->findAncestorOpInBlock(*useOp);
    // Check whether the use is in our block and after the current end
    // operation.
    if (useOp && endOperation->isBeforeInBlock(useOp))
      endOperation = useOp;
  }
  return endOperation;
}


LogicalResult deallocateBuffers(Operation *op) {
  OpBuilder builder(op);
  auto loc = op->getLoc();
  // ret val
  Value view = *op->result_begin();
  // find pos last user (assumes no copies, i.e. no control flow)
  if (view.getUsers().empty()) {
    op->emitWarning() << "trying to deallocate unused view";
    return success();
  }
  Operation *lastUser = getEndOperation(view, op);
  builder.setInsertionPointAfter(lastUser);
  builder.create<quiccir::DeallocOp>(loc, view);
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
//  QuiccirViewDeallocationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the quiccir operations
namespace {
struct  QuiccirViewDeallocationPass
    : public QuiccirViewDeallocationBase<QuiccirViewDeallocationPass> {
  void runOnOperation() override;
};
} // namespace

void  QuiccirViewDeallocationPass::runOnOperation() {

  // Walk from root func
  WalkResult result = getOperation()->walk([&](quiccir::AllocOp op) {
      // llvm::errs() << op.getOperationName() << '\n';
      // return deallocateBuffers(op);
      if (failed(deallocateBuffers(op)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

  if (result.wasInterrupted())
    signalPassFailure();
}

/// Create a pass for lowering operations to library calls
std::unique_ptr<Pass> mlir::quiccir::createViewDeallocationPass() {
  return std::make_unique<QuiccirViewDeallocationPass>();
}
