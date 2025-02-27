//====- TransposeGrouping.cpp - Group Transpose ops ---------------===//
//
// This file implements a Grouping of n (group option) akin transpose.
//
//===----------------------------------------------------------------------===//


#include "Quiccir/Transforms/QuiccirPassDetail.h"
#include "Quiccir/Transforms/QuiccirPasses.h"

#include "Quiccir/IR/QuiccirDialect.h"
#include "Quiccir/IR/QuiccirOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include <queue>

using namespace mlir;
using namespace mlir::quiccir;

namespace mlir::quiccir {
#define GEN_PASS_DEF_QUICCIRTRANSPOSEGROUPING
#include "Quiccir/Transforms/QuiccirPasses.h.inc"
} // namespace mlir::quiccir

namespace {

bool isSameTranspose(TransposeOp lhsOp, TransposeOp rhsOp) {
  // Check if the permutations are the same
  auto checkPerm = [](const llvm::ArrayRef<int64_t> lhsPerm,
                      const llvm::ArrayRef<int64_t> rhsPerm) {
    if (lhsPerm.size() != rhsPerm.size()) {
      return false;
    }
    for (std::size_t i = 0; i < lhsPerm.size(); i++) {
      if (lhsPerm[i] != rhsPerm[i]) {
        return false;
      }
    }
    return true;
  };
  bool isSamePemutation =
      checkPerm(lhsOp.getPermutation(), rhsOp.getPermutation());

  // Check if the operands are of the same space (projection level)
  auto checkSpace = [](Value lhs, Value rhs) {
    auto lhsOp = lhs.getDefiningOp();
    auto rhsOp = rhs.getDefiningOp();
    if (lhsOp == nullptr || rhsOp == nullptr) {
      // There is no defining op, must be a func arg
      // then we assume that are the same space
      // if they have the same type
      auto checkType = [](RankedTensorType lhsType, RankedTensorType rhsType) {
        return lhsType == rhsType;
      };
      return checkType(lhs.getType().cast<RankedTensorType>(),
                       rhs.getType().cast<RankedTensorType>());
    }
    return lhsOp->getName() == rhsOp->getName();
  };
  bool isSameSpace = checkSpace(lhsOp.getInput()[0], rhsOp.getInput()[0]);
  return isSamePemutation && isSameSpace;
}


/// Fix the use def chain using BFS
/// \return true if the dominance is fixed
bool fixDominance(Operation* op) {
  std::queue<Operation*> bfsQueue;
  bfsQueue.push(op);
  bool somethingChanged = false;
  while (!bfsQueue.empty()) {
    Operation* currentOp = bfsQueue.front();
    bfsQueue.pop();
    for (Value operand : currentOp->getOperands()) {
      Operation* defOp = operand.getDefiningOp();
      if (defOp) {
        if (!defOp->isBeforeInBlock(currentOp)) {
          somethingChanged = true;
          defOp->moveBefore(currentOp);
        }
        bfsQueue.push(defOp);
      }
    }
  }
  return somethingChanged;
};


//===----------------------------------------------------------------------===//
// TransposeGrouping over func ops
//===----------------------------------------------------------------------===//
class TransposeGrouping : public OpRewritePattern<func::FuncOp> {
private:
  int32_t group;

public:
  TransposeGrouping(MLIRContext *ctx, int32_t group)
      : OpRewritePattern<func::FuncOp>(ctx, /*benefit=*/1), group(group){};

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const final {

    // Walk from root func and collect transposes to be grouped
    SmallVector<Operation *, 4> transposeOps;
    bool needToGroup = false;
    WalkResult result = funcOp.walk([&](Operation *op) {
      // Get the first transpose op that is not grouped
      if (auto transposeOp = dyn_cast<TransposeOp>(op)) {
        // Check if the transpose op has more than one result
        if (transposeOp->getNumResults() > 1) {
          // Skip
          return WalkResult::advance();
        } else {
          // Is this the first transpose with a single result?
          if (transposeOps.empty()) {
            // Then store it
            transposeOps.push_back(transposeOp);
            return WalkResult::advance();
          } else {
            // Check if the transpose op is the same as the collected ones
            if (isSameTranspose(dyn_cast<TransposeOp>(transposeOps.front()),
                                transposeOp)) {
              // Collect
              needToGroup = true;
              transposeOps.push_back(transposeOp);
              // Stop if we have collected enough transposes
              if (static_cast<int>(transposeOps.size()) == group) {
                return WalkResult::interrupt();
              }
            }
          }
        }
      }
      return WalkResult::advance();
    });

    if (needToGroup) {
      // Group the collected transposes

      // Collect inputs and return types
      SmallVector<Value, 4> inputs;
      SmallVector<Type, 4> resultTypes;
      for (auto transposeOp : transposeOps) {
        inputs.push_back(cast<TransposeOp>(transposeOp).getInput()[0]);
        resultTypes.push_back(
            cast<TransposeOp>(transposeOp).getResult(0).getType());
      }

      // Set rewriter and insertion point
      rewriter.setInsertionPoint(transposeOps.front());

      // Create a new transpose op
      // the location needs to be the same as the last transpose op
      // otherwise the the operands might not dominate their uses
      rewriter.startRootUpdate(funcOp);
      auto newTranspose = rewriter.create<TransposeOp>(
          transposeOps.front()->getLoc(), resultTypes, inputs,
          cast<TransposeOp>(transposeOps.front()).getPermutation(),
          ::mlir::IntegerAttr{});

      // Replace the old transpose uses with the new transpose values
      for (std::size_t i = 0; i < transposeOps.size(); i++) {
        transposeOps[i]->getResult(0).replaceAllUsesWith(
            newTranspose.getResult(i));
        // Erase the old transpose op
        rewriter.eraseOp(transposeOps[i]);
      }

      // We need to fix the dominance of func body
      /// \todo to generalize to funcs with cfg
      /// change the grouping to be block based
      for (Block &block : funcOp.getBlocks()) {
        bool isBeingReordered = false;
        do {
          // The operands of the block must post dominate
          // their definitions
          Operation *terminator = block.getTerminator();
          isBeingReordered = fixDominance(terminator);
        } while (isBeingReordered);
      }

      // Rewriting is done
      rewriter.finalizeRootUpdate(funcOp);
      return success();
    }
    return failure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// QuiccirTransposeGroupingPass
//===----------------------------------------------------------------------===//

/// This is a rewrite pass
namespace {
struct QuiccirTransposeGroupingPass
    : public quiccir::impl::QuiccirTransposeGroupingBase<
          QuiccirTransposeGroupingPass> {
  using QuiccirTransposeGroupingBase<
      QuiccirTransposeGroupingPass>::QuiccirTransposeGroupingBase;
  void runOnOperation() final;
};

} // namespace

void QuiccirTransposeGroupingPass::runOnOperation() {
  if (group == 1) {
    getOperation()->emitError(
        "Group option must be greater than 1 or negative to express group all");
    signalPassFailure();
    return;
  }

  RewritePatternSet patterns(&getContext());
  patterns.add<TransposeGrouping>(&getContext(), group);

  FrozenRewritePatternSet patternSet(std::move(patterns));
  if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
    signalPassFailure();
}

/// Create a pass for grouping akin transpose ops
std::unique_ptr<Pass> mlir::quiccir::createTransposeGroupingPass() {
  return std::make_unique<QuiccirTransposeGroupingPass>();
}

std::unique_ptr<Pass> mlir::quiccir::createTransposeGroupingPass(
    const QuiccirTransposeGroupingOptions &options) {
  return std::make_unique<QuiccirTransposeGroupingPass>(options);
}
