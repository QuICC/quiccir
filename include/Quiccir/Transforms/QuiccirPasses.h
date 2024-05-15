//===- QuiccirPasses.h - Quiccir dialect passes -----------------*- C++ -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#ifndef QUICCIR_TRANSFORMS_QUICCIRPASSES_H
#define QUICCIR_TRANSFORMS_QUICCIRPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace quiccir {

/// Generate the code for options
#define GEN_PASS_DECL_QUICCIRVIEWWRAPPER
#include "Quiccir/Transforms/QuiccirPasses.h.inc"

/// Create a pass for adding quiccir deallocation ops
std::unique_ptr<mlir::Pass> createViewDeallocationPass();

/// Create a pass for lowering to quiccir operations to call op
std::unique_ptr<mlir::Pass> createLowerToCallPass();

/// Create a pass for lowering alloc op
std::unique_ptr<mlir::Pass> createLowerAllocPass();

/// Create a pass for lowering view in func/call/return/op
std::unique_ptr<mlir::Pass> createFinalizeViewToLLVMPass();

/// Create a pass for adding missing view layout info
static std::array<std::array<std::string, 2>, 3> defaultLayout;
std::unique_ptr<mlir::Pass> createSetViewLayoutPass(
    const std::array<std::array<std::string, 2>, 3> &layout = defaultLayout);

/// Create a pass for adding missing dimensions
std::unique_ptr<mlir::Pass> createSetDimensionsPass(
    llvm::ArrayRef<int64_t> phys = {},
    llvm::ArrayRef<int64_t> mods = {});

/// Create a pass for adding a view wrapper for entry point
static QuiccirViewWrapperOptions defaultViewWrapper;
std::unique_ptr<mlir::Pass> createViewWrapperPass(const QuiccirViewWrapperOptions &options = defaultViewWrapper);


//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Quiccir/Transforms/QuiccirPasses.h.inc"

} // namespace quiccir
} // namespace mlir

#endif // QUICCIR_TRANSFORMS_QUICCIRPASSES_H
