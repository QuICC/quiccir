//===- KindOpInterface.td - Shape Inference Interface -*- tablegen -==//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===----------------------------------------------------------------------===//
//
// Defines the operations that are in common to all Kind ops.
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_INTERFACES_KINDOPINTERFACE
#define QUICCIR_INTERFACES_KINDOPINTERFACE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def KindOpInterface : OpInterface<"KindOpInterface"> {
  let description = [{
    Interface for easy access to king attribute.
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

#endif // QUICCIR_INTERFACES_KINDOPINTERFACE
