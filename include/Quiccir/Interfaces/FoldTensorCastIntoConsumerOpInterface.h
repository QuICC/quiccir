//===- FoldTensorCastIntoConsumerOpInterface.h - Interface definitions for ShapeInference -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the shape inference interfaces defined
// in FoldTensorCastIntoConsumerOpInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_INTERFACES_FOLDTENSORCASTINTOCONSUMEROPINTERFACE_H
#define QUICCIR_INTERFACES_FOLDTENSORCASTINTOCONSUMEROPINTERFACE_H

#include "mlir/IR/OpDefinition.h"

/// Include the auto-generated declarations.
#include "Quiccir/Interfaces/FoldTensorCastIntoConsumerOpInterface.h.inc"

#endif // QUICCIR_INTERFACES_FOLDTENSORCASTINTOCONSUMEROPINTERFACE_H
