//===- ShapeInferenceOpInterface.td - Shape Inference Interface -*- tablegen -==//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Shape Inference Op Interface.
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_INTERFACES_SHAPEINFERENCEOPINTERFACE
#define QUICCIR_INTERFACES_SHAPEINFERENCEOPINTERFACE

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInferenceOpInterface"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let cppNamespace = "::mlir::quiccir";

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}

#endif // QUICCIR_INTERFACES_SHAPEINFERENCEOPINTERFACE
