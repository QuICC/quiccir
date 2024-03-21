//===- QuiccirOps.cpp - Quiccir dialect ops ---------------------*- C++ -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#include "Quiccir/IR/QuiccirOps.h"
#include "Quiccir/IR/QuiccirDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "canonicalizer"

#define GET_OP_CLASSES
#include "Quiccir/IR/QuiccirOps.cpp.inc"

/// Include the auto-generated definitions for the shape inference interfaces.
#include "Quiccir/Interfaces/ShapeInferenceOpInterface.cpp.inc"

using namespace mlir;
using namespace mlir::quiccir;

//===----------------------------------------------------------------------===//
// FrPOp
//===----------------------------------------------------------------------===//

/// \todo this is not correct we could come up with some
/// rule, for instance min to dealias but this should be an input.
/// we should have a set grid pass for ops that cannot be inferred
// void FrPOp::inferShapes() { getResult().setType(getMods().getType()); }

// LogicalResult mlir::quiccir::FrPOp::canonicalize(mlir::quiccir::FrPOp frOp,
//   PatternRewriter &rewriter) {

//   // Ask the operation to infer its output shapes.
//   Operation *op = frOp;
//   LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << op->getName() << '\n');
//   if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
//     shapeOp.inferShapes();


//   } else {
//     op->emitError("unable to infer shape of operation without shape "
//                   "inference interface");
//     return failure();
//   }


//   return success();
// }
