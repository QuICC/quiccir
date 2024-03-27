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
        Op<Quiccir_Dialect, mnemonic, traits> {

    string inferTransformShapes = [{
        void $cppClass::inferShapes() {
            Type in = (*getODSOperands(0).begin()).getType();
            Type out = getResult().getType();

            auto inType = llvm::dyn_cast<RankedTensorType>(in);
            auto outType = llvm::dyn_cast<RankedTensorType>(out);

            // Requires RankedTensorType.
            if (!inType || !outType)
                return;

            llvm::ArrayRef<int64_t> inShape = inType.getShape();
            llvm::ArrayRef<int64_t> outShape = outType.getShape();

            // Try to propagate input
            SmallVector<int64_t, 3> newOutShape{outShape};
            for (auto idx : {0, 2}) {
                if (outType.isDynamicDim(idx) && !inType.isDynamicDim(idx)) {
                    newOutShape[idx] = inShape[idx];
                }
            }
            getResult().setType(outType.clone(newOutShape));

            // Try to propagate output
            SmallVector<int64_t, 3> newInShape{inShape};
            for (auto idx : {0, 2}) {
                if (!outType.isDynamicDim(idx) && inType.isDynamicDim(idx)) {
                    newInShape[idx] = outShape[idx];
                }
            }
            (*getODSOperands(0).begin()).setType(inType.clone(newInShape));
        }
    }];

}

#endif // QUICCIR_DIALECT
