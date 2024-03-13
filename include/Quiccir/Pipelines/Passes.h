//===- Passes.h - Quiccir dialect pipelines --------------------*- C++ -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#ifndef QUICCIR_PIPELINES_PASSES_H
#define QUICCIR_PIPELINES_PASSES_H

#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace quiccir {

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Build pipeline to lower to QuICC library calls.
void quiccLibCallPipelineBuilder(OpPassManager &pm);

/// Registers all pipelines for the `sparse_tensor` dialect.  At present,
/// this includes only "sparse-compiler".
void registerQuiccirPipelines();

} // namespace quiccir
} // namespace mlir

#endif // QUICCIR_PIPELINES_PASSES_H
