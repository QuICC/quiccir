//===- QuiccirDialect.cpp - Quiccir dialect ---------------------*- C++ -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#include "Quiccir/IR/QuiccirDialect.h"
#include "Quiccir/IR/QuiccirOps.h"

// #include "mlir/IR/Builders.h"
// #include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::quiccir;

#include "Quiccir/IR/QuiccirOpsDialect.cpp.inc"

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
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within Quiccir can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  // All functions within Quiccir can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<tensor::CastOp>(conversionLoc, resultType, input);
  }
};

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult DeallocOp::verify() {
  // Check that this is the last use of the operand
  Value view = getOperand();
  Operation *lastUser = *(view.user_begin()); // single link list
  if (lastUser != *this) {
    return lastUser->emitError()
      << "found uses of dealloc operand";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// MaterializeOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult MaterializeOp::verify() {
  auto tensor = getOperand(0);
  auto tensorType = tensor.getType().dyn_cast<RankedTensorType>();
  auto view = getOperand(1);
  auto viewType = view.getType().dyn_cast<ViewType>();

  // Check rank
  if (tensorType.getRank() != viewType.getRank()) {
    return emitOpError()
      << "rank mismatch, tensor="
      << tensorType.getRank()
      << " while view="
      << viewType.getRank();
  }

  // Check size
  if (tensorType.getShape() != viewType.getShape()) {
    return emitOpError()
      << "shape mismatch, tensor="
      << tensorType.getShape()
      << " while view="
      << viewType.getShape();
  }

  // Check element type
  if (tensorType.getElementType() != viewType.getElementType()) {
    return emitOpError()
      << "type mismatch, tensor="
      << tensorType.getElementType()
      << " while view="
      << viewType.getElementType();
  }

  return mlir::success();
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

  // If unranked, there is nothing to check
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
           << "operand #2 expected rank=3 instead rank="
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
           << "expected opTensor third dimension to match umod second dimension";
  }
  if (opTensorShape[0] != umodShape[0]) {
    return emitError()
           << "expected opTensor first dimension to match umod first dimension";
  }
  auto uvalShape = resultType.getShape();
  if (opTensorShape[0] != uvalShape[0]) {
    return emitError()
           << "expected result first dimension to match umod first dimension";
  }
  if (opTensorShape[1] != uvalShape[1]) {
    return emitError()
           << "expected result second dimension to opTensor second dimension";
  }
  if (umodShape[2] != uvalShape[2]) {
    return emitError()
           << "expected result third dimension to umod third dimension";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// FrPOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult FrPOp::verify() {
  auto mod = getOperand();
  auto modType = mod.getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();

  // If unranked, there is nothing to check
  if (!modType || !resultType)
    return mlir::success();

  // Check ranks
  if (modType.getRank() != 3) {
    return emitOpError()
           << "operand #1 expected rank=3 instead rank="
           << modType.getRank();
  }
  if (resultType.getRank() != 3) {
    return emitOpError()
           << "return value expected rank=3 instead rank="
           << resultType.getRank();
  }

  // Check consistency of the number of modes
  // right most is first logical
  auto modShape = modType.getShape();
  auto valShape = resultType.getShape();

  if ((valShape[0] != ShapedType::kDynamic &&
       modShape[0] != ShapedType::kDynamic) &&
      modShape[0] != valShape[0]) {
    return emitError()
      << "expected result first dimension " << valShape[0]
      << "to match mod first dimension " << modShape[0];
  }
  if ((valShape[2] != ShapedType::kDynamic &&
       modShape[2] != ShapedType::kDynamic) &&
      modShape[2] != valShape[2]) {
    return emitError()
      << "expected result third dimension " << valShape[2]
      << "to match mod third dimension " << modShape[2];
  }

  // Todo: check dim attribute consistency if available

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// FrIOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult FrIOp::verify() {
  auto phys = getOperand();
  auto physType = phys.getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();

  // If unranked, there is nothing to check
  if (!physType || !resultType)
    return mlir::success();

  // Check ranks
  if (physType.getRank() != 3) {
    return emitOpError()
           << "operand #1 expected rank=3 instead rank="
           << physType.getRank();
  }
  if (resultType.getRank() != 3) {
    return emitOpError()
           << "return value expected rank=3 instead rank="
           << resultType.getRank();
  }

  // Check consistency of the number of integration points
  // right most is first logical
  auto physShape = physType.getShape();
  auto valShape = resultType.getShape();

  if ((valShape[0] != ShapedType::kDynamic &&
       physShape[0] != ShapedType::kDynamic) &&
      physShape[0] != valShape[0]) {
    return emitError()
      << "expected result first dimension " << valShape[0]
      << "to match phys first dimension " << physShape[0];
  }
  if ((valShape[2] != ShapedType::kDynamic &&
       physShape[2] != ShapedType::kDynamic) &&
      physShape[2] != valShape[2]) {
    return emitError()
      << "expected result third dimension " << valShape[2]
      << "to match phys third dimension " << physShape[2];
  }

  // Todo: check dim attribute consistency if available

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AlPOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AlPOp::verify() {
  auto mod = getOperand();
  auto modType = mod.getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();

  // If unranked, there is nothing to check
  if (!modType || !resultType)
    return mlir::success();

  // Check ranks
  if (modType.getRank() != 3) {
    return emitOpError()
           << "operand #1 expected rank=3 instead rank="
           << modType.getRank();
  }
  if (resultType.getRank() != 3) {
    return emitOpError()
           << "return value expected rank=3 instead rank="
           << resultType.getRank();
  }

  // Check consistency of the number of modes
  // right most is first logical
  auto modShape = modType.getShape();
  auto valShape = resultType.getShape();
  if (modShape[0] != valShape[0]) {
    return emitError()
           << "expected result first dimension to match mod first dimension";
  }
  if (modShape[2] != valShape[2]) {
    return emitError()
           << "expected result third dimension to match mod third dimension";
  }

  // Todo: check dim attribute consistency if available

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AlIOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AlIOp::verify() {
  auto phys = getOperand();
  auto physType = phys.getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();

  // If unranked, there is nothing to check
  if (!physType || !resultType)
    return mlir::success();

  // Check ranks
  if (physType.getRank() != 3) {
    return emitOpError()
           << "operand #1 expected rank=3 instead rank="
           << physType.getRank();
  }
  if (resultType.getRank() != 3) {
    return emitOpError()
           << "return value expected rank=3 instead rank="
           << resultType.getRank();
  }

  // Check consistency of the number of integration points
  // right most is first logical
  auto physShape = physType.getShape();
  auto valShape = resultType.getShape();
  if (physShape[0] != valShape[0]) {
    return emitError()
           << "expected result first dimension to match phys first dimension";
  }
  if (physShape[2] != valShape[2]) {
    return emitError()
           << "expected result third dimension to match phys third dimension";
  }

  // Todo: check dim attribute consistency if available

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// JWPOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult JWPOp::verify() {
  auto mod = getOperand();
  auto modType = mod.getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();

  // If unranked, there is nothing to check
  if (!modType || !resultType)
    return mlir::success();

  // Check ranks
  if (modType.getRank() != 3) {
    return emitOpError()
           << "operand #1 expected rank=3 instead rank="
           << modType.getRank();
  }
  if (resultType.getRank() != 3) {
    return emitOpError()
           << "return value expected rank=3 instead rank="
           << resultType.getRank();
  }

  // Check consistency of the number of modes
  // right most is first logical
  auto modShape = modType.getShape();
  auto valShape = resultType.getShape();
  if (modShape[0] != valShape[0]) {
    return emitError()
           << "expected result first dimension to match mod first dimension";
  }
  if (modShape[2] != valShape[2]) {
    return emitError()
           << "expected result third dimension to match mod third dimension";
  }

  // Todo: check dim attribute consistency if available

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// JWIOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult JWIOp::verify() {
  auto phys = getOperand();
  auto physType = phys.getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();

  // If unranked, there is nothing to check
  if (!physType || !resultType)
    return mlir::success();

  // Check ranks
  if (physType.getRank() != 3) {
    return emitOpError()
           << "operand #1 expected rank=3 instead rank="
           << physType.getRank();
  }
  if (resultType.getRank() != 3) {
    return emitOpError()
           << "return value expected rank=3 instead rank="
           << resultType.getRank();
  }

  // Check consistency of the number of integration points
  // right most is first logical
  auto physShape = physType.getShape();
  auto valShape = resultType.getShape();
  if (physShape[0] != valShape[0]) {
    return emitError()
           << "expected result first dimension to match phys first dimension";
  }
  if (physShape[2] != valShape[2]) {
    return emitError()
           << "expected result third dimension to match phys third dimension";
  }

  // Todo: check dim attribute consistency if available

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// QuiccirDialect
//===----------------------------------------------------------------------===//

void QuiccirDialect::initialize() {
  /// Add the defined operations in the dialect.
  addOperations<
#define GET_OP_LIST
#include "Quiccir/IR/QuiccirOps.cpp.inc"
      >();

  addInterfaces<QuiccirInlinerInterface>();

  registerTypes();
}
