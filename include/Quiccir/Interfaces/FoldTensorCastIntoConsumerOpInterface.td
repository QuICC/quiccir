//===- FoldTensorCastIntoConsumerOpInterface.td - Fold Cast Interface -*- tablegen -==//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//
//
// Defines the operations of the operations that can fold a tensor producer.
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_INTERFACES_FOLDTENSORCASTINTOCONSUMEROPINTERFACE
#define QUICCIR_INTERFACES_FOLDTENSORCASTINTOCONSUMEROPINTERFACE

include "mlir/IR/OpBase.td"

def FoldTensorCastIntoConsumerOpInterface :
    OpInterface<"FoldTensorCastIntoConsumerOpInterface"> {
  let description = [{
    Interface to mark ops that can fold a producer tensor.cast.
  }];

  let cppNamespace = "::mlir::quiccir";
}

#endif // QUICCIR_INTERFACES_FOLDTENSORCASTINTOCONSUMEROPINTERFACE
