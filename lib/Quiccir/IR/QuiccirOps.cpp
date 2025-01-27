//===- QuiccirOps.cpp - Quiccir dialect ops ---------------------*- C++ -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#include "Quiccir/IR/QuiccirOps.h"
#include "Quiccir/IR/QuiccirDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "canonicalizer"

#define GET_OP_CLASSES
#include "Quiccir/IR/QuiccirOps.cpp.inc"

/// Include the auto-generated definitions for the interfaces.
#include "Quiccir/Interfaces/KindOpInterface.cpp.inc"
#include "Quiccir/Interfaces/ShapeInferenceOpInterface.cpp.inc"
#include "Quiccir/Interfaces/TransformOpInterface.cpp.inc"

using namespace mlir;
using namespace mlir::quiccir;

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//
/// \todo fix infer, need to check for attributes
void AddOp::inferShapes() {
  if (tensor::preservesStaticInformation(getLhs().getType(),
                                         getResult().getType())) {
    getLhs().setType(getResult().getType());
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Result has less info then Lhs\n");
  }
  if (tensor::preservesStaticInformation(getRhs().getType(),
                                         getResult().getType())) {
    getRhs().setType(getResult().getType());
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Result has less info then Rhs\n");
  }
  if (tensor::preservesStaticInformation(getResult().getType(),
                                         getLhs().getType())) {
    getResult().setType(getLhs().getType());
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Lhs has less info then result\n");
  }
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//
/// \todo fix infer, need to check for attributes
void SubOp::inferShapes() {
  if (tensor::preservesStaticInformation(getLhs().getType(),
                                         getResult().getType())) {
    getLhs().setType(getResult().getType());
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Result has less info then Lhs\n");
  }
  if (tensor::preservesStaticInformation(getRhs().getType(),
                                         getResult().getType())) {
    getRhs().setType(getResult().getType());
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Result has less info then Rhs\n");
  }
  if (tensor::preservesStaticInformation(getResult().getType(),
                                         getLhs().getType())) {
    getResult().setType(getLhs().getType());
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Lhs has less info then result\n");
  }
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
void TransposeOp::inferShapes() {
  auto inRange = getInput().getType();
  auto outRange = getOutput().getType();

  // This is checked by the verifier.
  assert(inRange.size() == outRange.size() && "Input and output ranges must have the same size");
  for (std::size_t i = 0; i < inRange.size(); ++i) {
    auto inType = inRange[i].dyn_cast<RankedTensorType>();
    auto outType = outRange[i].dyn_cast<RankedTensorType>();

    // Requires RankedTensorType.
    if (!inType || !outType)
      continue;

    llvm::ArrayRef<int64_t> inShape = inType.getShape();
    llvm::ArrayRef<int64_t> outShape = outType.getShape();

    // Try to propagate input
    auto perm = getPermutation();
    SmallVector<int64_t, 3> newOutShape{outShape};
    constexpr std::array<int, 3> indices = {0, 1, 2};
    for (auto idx : indices) {
      if (outType.isDynamicDim(perm[idx]) && !inType.isDynamicDim(idx)) {
        newOutShape[perm[idx]] = inShape[idx];
      }
    }
    getResult(i).setType(outType.clone(newOutShape));

    // Try to propagate output
    SmallVector<int64_t, 3> newInShape{inShape};
    for (auto idx : indices) {
      if (!outType.isDynamicDim(perm[idx]) && inType.isDynamicDim(idx)) {
        newInShape[idx] = outShape[perm[idx]];
      }
    }
    getInput()[i].setType(inType.clone(newInShape));
  }
}
