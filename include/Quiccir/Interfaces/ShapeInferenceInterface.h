//===- ShapeInferenceInterface.h - Interface definitions for ShapeInference -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_INTERFACES_SHAPEINFERENCEINTERFACE_H_
#define QUICCIR_INTERFACES_SHAPEINFERENCEINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace quiccir {

/// Include the auto-generated declarations.
#include "Quiccir/Interfaces/ShapeInferenceInterface.h.inc"

} // namespace quiccir
} // namespace mlir

#endif // QUICCIR_INTERFACES_SHAPEINFERENCEINTERFACE_H_
