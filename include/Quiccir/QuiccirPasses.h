//===- QuiccirPasses.h - Quiccir dialect passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_QUICCIRPASSES_H
#define QUICCIR_QUICCIRPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace quiccir {

/// Create a pass for adding quiccir deallocation ops
std::unique_ptr<mlir::Pass> createViewDeallocationPass();

/// Create a pass for lowering to quiccir operations to call op
std::unique_ptr<mlir::Pass> createLowerToCallPass();

/// Create a pass for lowering alloc op
std::unique_ptr<mlir::Pass> createLowerAllocPass();

/// Create a pass for lowering view in func/call/return/op
std::unique_ptr<mlir::Pass> createFinalizeViewToLLVMPass();

// /// Create a pass for lowering to operations in the `Affine` and `Std` dialects
// std::unique_ptr<mlir::Pass> createLowerToAffinePass();

// /// Create a pass for lowering to operations in the `Linalg` and `Affine` dialects
// std::unique_ptr<mlir::Pass> createLowerToLinalgPass();



//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Quiccir/QuiccirPasses.h.inc"

} // namespace quiccir
} // namespace mlir

#endif // QUICCIR_QUICCIRPASSES_H
