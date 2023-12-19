//===- QuiccirOps.td - Quiccir dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_OPS
#define QUICCIR_OPS

include "Quiccir/QuiccirDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// QuadratureOp
//===----------------------------------------------------------------------===//

def Quiccir_QuadratureOp : Quiccir_Op<"quadrature", [Pure]> {
    let summary = "Batched backward transform kernel";
    let description = [{
        The `quiccir.quadrature` operation implements a batched quadrature based transform kernel.
        At the moment this is a dense operation, next should be implemented as sparse.

        This operation take 2 tensors rapresenting, one representing the operator the second the
        modes/quadrature values, it returns a tensor representing the value at the
        integration points or the modes.

        The shape of the operator and the input tensor need to fit.

        The operation represents op * mods = phys or op * vals = mods.

        Example:

        ```mlir
        %uphys = quiccir.quadrature %op, %umods: tensor<?x?x?x?>, tensor<?x?x?x?> -> tensor<?x?x?x?>
        ```
    }];

    let arguments = (ins AnyTensor:$op, AnyTensor:$umods);
    let results = (outs AnyTensor:$uphys);

    let assemblyFormat = [{
        $op `,` $umods attr-dict `:`  type($op) `,` type($umods)  `->` type($uphys)
    }];

    // Add additional verification logic to the constant operation. Setting this bit
    // to `1` will generate a `::mlir::LogicalResult verify()` declaration on the
    // operation class that is called after ODS constructs have been verified, for
    // example the types of arguments and results. We implement additional verification
    // in the definition of this `verify` method in the C++ source file.
    let hasVerifier = 1;

    // Allow building a QuadratureOp from the three input operands.
    let builders = [
        OpBuilder<(ins "Value":$op, "Value":$umods)>
    ];
}

#endif // QUICCIR_OPS
