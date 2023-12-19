//===- QuiccirDialect.cpp - Quiccir dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Quiccir/QuiccirDialect.h"
#include "Quiccir/QuiccirOps.h"

// #include "mlir/IR/Builders.h"
// #include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::quiccir;

#include "Quiccir/QuiccirOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// QuiccirInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with Toy
/// operations.
struct QuiccirInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All operations within Quiccir can be inlined.
  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const final {
    return true;
  }

};

//===----------------------------------------------------------------------===//
// Quiccir dialect.
//===----------------------------------------------------------------------===//

void QuiccirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Quiccir/QuiccirOps.cpp.inc"
      >();

  addInterfaces<QuiccirInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// QuadratureOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult QuadratureOp::verify() {
  auto opTensor = getOperand(0);
  auto opTensorType = opTensor.getType().dyn_cast<RankedTensorType>();
  auto umod = getOperand(1);
  auto umodType = umod.getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();
  if (!opTensorType || !umodType || !resultType)
    return mlir::success();

  // Check ranks
  if (opTensorType.getRank() != 3) {
    return emitOpError()
           << "operand #1 expected rank=3 instead rank="
           << opTensorType.getRank();
  }
  if (umodType.getRank() != 3) {
    return emitOpError()
           << "operand #3 expected rank=3 instead rank="
           << umodType.getRank();
  }
  if (resultType.getRank() != 3) {
    return emitOpError()
           << "return value expected rank=3 instead rank="
           << resultType.getRank();
  }

  // Check consistency of the number of modes
  // right most is first logical
  auto opTensorShape = opTensorType.getShape();
  auto umodShape = umodType.getShape();
  if (opTensorShape[2] != umodShape[1]) {
    return emitError()
           << "expected opTensor second dimension to match umod first dimension";
  }
  if (opTensorShape[0] != umodShape[0]) {
    return emitError()
           << "expected opTensor third dimension to match umod third dimension";
  }
  return mlir::success();
}
