//===- QuiccirDialect.td - Quiccir dialect -----------------*- tablegen -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#ifndef QUICCIR_DIALECT
#define QUICCIR_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Quiccir dialect definition.
//===----------------------------------------------------------------------===//

def Quiccir_Dialect : Dialect {
    let name = "quiccir";
    let summary = "A out-of-tree MLIR dialect to model QuICC operators.";
    let description = [{
        This dialect is an out-of-tree MLIR dialect (https://github.com/QuICC/quiccir) designed to
        model QuICC operators (https://github.com/QuICC/QuICC).
    }];
    let cppNamespace = "::mlir::quiccir";

    let useDefaultTypePrinterParser = 1;
    let hasCanonicalizer = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base quiccir operation definition.
//===----------------------------------------------------------------------===//

class Quiccir_Op<string mnemonic, list<Trait> traits = []> :
        Op<Quiccir_Dialect, mnemonic, traits>;

#endif // QUICCIR_DIALECT
