//===- QuiccirPassDetail.h - Quiccir dialect passes -------------*- C++ -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#ifndef QUICCIR_TRANSFORMS_QUICCIRPASSDETAIL_H
#define QUICCIR_TRANSFORMS_QUICCIRPASSDETAIL_H

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#include "Quiccir/IR/QuiccirDialect.h"

namespace mlir {
namespace quiccir {

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

/// Generate the code for passes base class.
#define GEN_PASS_CLASSES
#include "Quiccir/Transforms/QuiccirPasses.h.inc"

} // namespace quiccir
} // namespace mlir

#endif // QUICCIR_TRANSFORMS_QUICCIRPASSDETAIL_H
