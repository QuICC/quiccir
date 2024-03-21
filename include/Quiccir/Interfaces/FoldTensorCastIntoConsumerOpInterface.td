//===- FoldTensorCastIntoConsumerOpInterface.td - Shape Inference Interface -*- tablegen -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
