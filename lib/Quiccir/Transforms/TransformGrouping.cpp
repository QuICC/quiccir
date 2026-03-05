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
  bool isSamePermutation =
      checkPerm(lhsOp.getPermutation(), rhsOp.getPermutation());

  // Check if the operands are of the same space (projection level)
  auto checkSpace = [](Value lhs, Value rhs) {
    /// \todo add transform/projection interface
    auto isTransform = [](Operation *op) {
      if (op == nullptr) {
        return false;
      }
      return isa<FrIOp>(op) || isa<FrPOp>(op) || isa<AlIOp>(op) || isa<AlIVOp>(op) ||
             isa<AlPOp>(op) || isa<AlPVOp>(op) || isa<JWIOp>(op) || isa<JWPOp>(op);
    };
    // Iteratively go up the defining op chain
    // until we reach either a func arg or a transpose op
    auto findOpLevel = [&](Operation *op) -> auto {
      while (op != nullptr) {
        if (isTransform(op)) {
          // We are done
          break;
        }
        op = op->getOperand(0).getDefiningOp();
      }
      return op;
    };
    Operation *lhsOp = findOpLevel(lhs.getDefiningOp());
    Operation *rhsOp = findOpLevel(rhs.getDefiningOp());
    if (lhsOp == nullptr || rhsOp == nullptr) {
      // We could not find a relevant defining op, value must be a func arg
      // then we assume that are the same space if they have the same type
      auto checkType = [](RankedTensorType lhsType, RankedTensorType rhsType) {
        return lhsType == rhsType;
      };
      return checkType(lhs.getType().cast<RankedTensorType>(),
                       rhs.getType().cast<RankedTensorType>());
    }
    return lhsOp->getName() == rhsOp->getName();
  };
  bool isSameSpace = checkSpace(lhsOp.getInput()[0], rhsOp.getInput()[0]);
  return isSamePermutation && isSameSpace;
}

/// Fix the use def chain using BFS in a block
/// \return true if the SSA dominance is fixed
bool fixDominance(Block &block) {
  // The operands of the block terminator must post
  // dominate their definitions
  Operation *terminator = block.getTerminator();
  std::queue<Operation *> bfsQueue;
  bfsQueue.push(terminator);
  bool somethingChanged = false;
  while (!bfsQueue.empty()) {
    Operation *currentOp = bfsQueue.front();
    bfsQueue.pop();
    for (Value operand : currentOp->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
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

    // Collect transpose ops in a block as candidates for the grouping
    SmallVector<TransposeOp, 4> candidateOps;
    for (Block &block : funcOp.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        if (auto transposeOp = dyn_cast<TransposeOp>(op)) {
          // Check if the transpose op has only one result
          if (transposeOp->getNumResults() == 1) {
            candidateOps.push_back(transposeOp);
          }
        }
      }

      // Loop over candidates and store if same
      SmallVector<Operation *, 4> transposeOps;
      for (std::size_t i = 0; i < candidateOps.size(); i++) {
        transposeOps.push_back(candidateOps[i]);
        // Check if the transpose op is the same as the collected ones
        for (std::size_t j = i + 1; j < candidateOps.size(); j++) {
          if (isSameTranspose(candidateOps[i], candidateOps[j])) {
            // Collect
            transposeOps.push_back(candidateOps[j]);
            // Stop if we have collected enough transposes
            if (static_cast<int>(transposeOps.size()) == group) {
              break;
            }
          }
        }
        // If we have more than one transpose, we can group them
        if (transposeOps.size() > 1) {
          break;
        } else {
          transposeOps.clear();
        }
      }

      // Group the collected transposes
      if (transposeOps.size() > 1) {

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
        // we chose the first transpose op as the insertion point
        // later we will fix the ops dominance
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

        // We need to fix the ops SSA dominance in the block
        bool isBeingReordered = false;
        do {
          isBeingReordered = fixDominance(block);
        } while (isBeingReordered);

        // Rewriting is done
        rewriter.finalizeRootUpdate(funcOp);
        return success();
      }
    } // end of block loop
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
