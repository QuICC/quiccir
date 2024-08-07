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

def TransformOpInterface : OpInterface<"TransformOpInterface", [
  DeclareOpInterfaceMethods<FoldTensorCastIntoConsumerOpInterface>,
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
  ]> {
  let description = [{
    Interface to access methods that are in common to all transform ops.
  }];

  let cppNamespace = "::mlir::quiccir";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return Attribute kind as a sring.",
      /*retTy=*/"std::string",
      /*methodName=*/"getKind",
      /*args=*/(ins),
      /*methodBody*/[{
        auto kind = $_op.getKind();
        if (kind.has_value()) {
          return "_"+kind.value().str();
        }
        else {
          return "";
        }
      }]
    >
  ];
}

#endif // QUICCIR_INTERFACES_TRANSFORMOPINTERFACE
