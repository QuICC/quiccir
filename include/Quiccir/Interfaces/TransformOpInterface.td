//===- TransformOpInterface.td - Shape Inference Interface -*- tablegen -==//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===----------------------------------------------------------------------===//
//
// Defines the operations that are in common to all transform ops.
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_INTERFACES_TRANSFORMOPINTERFACE
#define QUICCIR_INTERFACES_TRANSFORMOPINTERFACE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Quiccir/Interfaces/ShapeInferenceOpInterface.td"
include "Quiccir/Interfaces/FoldTensorCastIntoConsumerOpInterface.td"
include "Quiccir/Interfaces/KindOpInterface.td"

def TransformOpInterface : OpInterface<"TransformOpInterface", [
  DeclareOpInterfaceMethods<FoldTensorCastIntoConsumerOpInterface>,
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
  DeclareOpInterfaceMethods<KindOpInterface>
  ]> {
  let description = [{
    Interface to access methods that are in common to all transform ops.
  }];

  let cppNamespace = "::mlir::quiccir";
}

#endif // QUICCIR_INTERFACES_TRANSFORMOPINTERFACE
