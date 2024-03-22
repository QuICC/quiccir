//===- QuiccirOps.cpp - Quiccir dialect ops ---------------------*- C++ -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#include "Quiccir/IR/QuiccirOps.h"
#include "Quiccir/IR/QuiccirDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "canonicalizer"

#define GET_OP_CLASSES
#include "Quiccir/IR/QuiccirOps.cpp.inc"

/// Include the auto-generated definitions for the shape inference interfaces.
#include "Quiccir/Interfaces/ShapeInferenceOpInterface.cpp.inc"

using namespace mlir;
using namespace mlir::quiccir;

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//
void AddOp::inferShapes() {
    if (tensor::preservesStaticInformation(getLhs().getType(),
        getResult().getType())) {
        getLhs().setType(getResult().getType());
    }
    else {
        LLVM_DEBUG(llvm::dbgs() << "Result has less info then Lhs\n");
    }
    if (tensor::preservesStaticInformation(getRhs().getType(),
        getResult().getType())) {
        getRhs().setType(getResult().getType());
    }
    else {
        LLVM_DEBUG(llvm::dbgs() << "Result has less info then Rhs\n");
    }
    if (tensor::preservesStaticInformation(getResult().getType(),
        getLhs().getType())) {
        getResult().setType(getLhs().getType());
    }
    else {
        LLVM_DEBUG(llvm::dbgs() << "Lhs has less info then result\n");
    }
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//
void SubOp::inferShapes() {
    if (tensor::preservesStaticInformation(getLhs().getType(),
        getResult().getType())) {
        getLhs().setType(getResult().getType());
    }
    else {
        LLVM_DEBUG(llvm::dbgs() << "Result has less info then Lhs\n");
    }
    if (tensor::preservesStaticInformation(getRhs().getType(),
        getResult().getType())) {
        getRhs().setType(getResult().getType());
    }
    else {
        LLVM_DEBUG(llvm::dbgs() << "Result has less info then Rhs\n");
    }
    if (tensor::preservesStaticInformation(getResult().getType(),
        getLhs().getType())) {
        getResult().setType(getLhs().getType());
    }
    else {
        LLVM_DEBUG(llvm::dbgs() << "Lhs has less info then result\n");
    }
}
