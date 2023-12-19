//===- QuiccirPassDetail.h - Quiccir dialect passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_QUICCIRPASSDETAIL_H
#define QUICCIR_QUICCIRPASSDETAIL_H

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace quiccir {

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

/// Generate the code for passes base class.
#define GEN_PASS_CLASSES
#include "Quiccir/QuiccirPasses.h.inc"

} // namespace quiccir
} // namespace mlir

#endif // QUICCIR_QUICCIRPASSDETAIL_H
