//===- ShapeInferenceInterface.h - Interface definitions for ShapeInference -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceOpInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_INTERFACES_SHAPEINFERENCEOPINTERFACE_H
#define QUICCIR_INTERFACES_SHAPEINFERENCEOPINTERFACE_H

#include "mlir/IR/OpDefinition.h"

/// Include the auto-generated declarations.
#include "Quiccir/Interfaces/ShapeInferenceOpInterface.h.inc"

#endif // QUICCIR_INTERFACES_SHAPEINFERENCEOPINTERFACE_H
