//===- TransformOpInterface.h - Interface definitions for Transform -=//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the transform interfaces defined
// in TransformOpInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_INTERFACES_TRANSFORMOPINTERFACE_H
#define QUICCIR_INTERFACES_TRANSFORMOPINTERFACE_H

#include "Quiccir/Interfaces/FoldTensorCastIntoConsumerOpInterface.h"
#include "Quiccir/Interfaces/KindOpInterface.h"
#include "Quiccir/Interfaces/ShapeInferenceOpInterface.h"
#include "mlir/IR/OpDefinition.h"

/// Include the auto-generated declarations.
#include "Quiccir/Interfaces/TransformOpInterface.h.inc"

#endif // QUICCIR_INTERFACES_TRANSFORMOPINTERFACE_H
