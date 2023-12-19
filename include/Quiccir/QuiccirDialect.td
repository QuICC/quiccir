//===- QuiccirDialect.td - Quiccir dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_DIALECT
#define QUICCIR_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Quiccir dialect definition.
//===----------------------------------------------------------------------===//

def Quiccir_Dialect : Dialect {
    let name = "quiccir";
    let summary = "A out-of-tree MLIR dialect to model nektar++ operators.";
    let description = [{
        This dialect is an out-of-tree MLIR dialect designed to
        model nektar++ operators (https://www.nektar.info/).
    }];
    let cppNamespace = "::mlir::quiccir";
    let useFoldAPI = kEmitFoldAdaptorFolder;
}

//===----------------------------------------------------------------------===//
// Base quiccir operation definition.
//===----------------------------------------------------------------------===//

class Quiccir_Op<string mnemonic, list<Trait> traits = []> :
        Op<Quiccir_Dialect, mnemonic, traits>;

#endif // QUICCIR_DIALECT
