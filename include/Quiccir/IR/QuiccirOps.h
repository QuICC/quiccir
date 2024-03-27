//===- QuiccirOps.h - Quiccir dialect ops -----------------------*- C++ -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#ifndef QUICCIR_IR_QUICCIROPS_H
#define QUICCIR_IR_QUICCIROPS_H

#include "Quiccir/IR/QuiccirTypes.h"
#include "Quiccir/Interfaces/ShapeInferenceOpInterface.h"
#include "Quiccir/Interfaces/FoldTensorCastIntoConsumerOpInterface.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Quiccir/IR/QuiccirOps.h.inc"

#endif // QUICCIR_IR_QUICCIROPS_H
