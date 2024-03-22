//===- QuiccirOps.td - Quiccir dialect ops -----------------*- tablegen -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#ifndef QUICCIR_OPS
#define QUICCIR_OPS

include "Quiccir/IR/QuiccirDialect.td"
include "Quiccir/IR/QuiccirTypes.td"
include "Quiccir/Interfaces/ShapeInferenceOpInterface.td"
include "Quiccir/Interfaces/FoldTensorCastIntoConsumerOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

def Quiccir_AllocOp : Quiccir_Op<"alloc", []> {
    let summary = "Allocate view";
    let description = [{
        The `quiccir.alloc` operation allocate a new view buffer.

        In order to allocate the right amount of space for the data
        and the meta data (and populate the metadata?) we need to
        know the producer kind and the input view.

        Example:

        ```mlir
        %0 = quiccir.alloc(%arg) : !quiccir.view<?x?x?xf64, "layout"> ->
            !quiccir.view<?x?x?xf64, "layout">  {producer="quiccir.op"}
        ```

        For codegen we should lower this to something like
        ```mlir
        %uphys_pos = memref.alloc() : memref<?xi32>
        %uphys_coo = memref.alloc() : memref<?xi32>
        %uphys_val = memref.alloc() : memref<?xf64>
        %uphys = quiccir.assemble(%uphys_pos, %uphys_coo, %uphys_val) : !quiccir.view<?x?x?xf64, "layout">
        ```
        or use directly the sparse tensor facilities

    }];

    let arguments = (ins AnyView:$producerView, StrAttr:$producer);
    let results = (outs AnyView:$view);

    let assemblyFormat = [{
        `(` $producerView `)` `:` qualified(type($producerView)) `->` qualified(type($view)) attr-dict
    }];

    // let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "ViewType":$viewType, "Value":$producerViewType, "StringAttr":$producerName), [{
            return build($_builder, $_state, viewType, producerViewType, producerName);
        }]>
    ];

    // let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

def Quiccir_DeallocOp : Quiccir_Op<"dealloc", []> {
    let summary = "Deallocate view";
    let description = [{
        The `quiccir.dealloc` operation deallocate a view buffer.

        Example:

        ```mlir
        quiccir.dealloc(%arg) : !quiccir.view<?x?x?xf64, "layout">
        ```
    }];

    let arguments = (ins AnyView:$view);

    let assemblyFormat = [{
        `(` $view `)` `:` qualified(type($view)) attr-dict
    }];

    // let skipDefaultBuilders = 1;
    // let builders = [
    //     OpBuilder<(ins "Value":$view), [{
    //         return build($_builder, $_state, view);
    //     }]>
    // ];

    let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// MaterializeOp
//===----------------------------------------------------------------------===//

def Quiccir_MaterializeOp : Quiccir_Op<"materialize", []> {
    let summary = "Materialize (sparse) tensor to view";
    let description = [{
        The `quiccir.materialize` operation materializes a tensor into an existing
        view buffer.
        At the moment this is a dense operation, next should be implemented as sparse.
        For codegen we should use the sparse tensor infrastructure.

        This operation takes one tensor and a view.
        In conjuction with the lowering of the tensor producer, it allows to reuse
        an existing buffer.

        Example:

        ```mlir
        quiccir.materialize %ret in %uval : (tensor<1x3x2xf64>, !quiccir.view<1x3x2xf64, "layout">)
        ```
    }];

    let arguments = (ins AnyRankedTensor:$tensor, AnyView:$view);

    let assemblyFormat = [{
        $tensor `in` $view  `:` `(` type($tensor) `,` qualified(type($view)) `)` attr-dict
    }];

    let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// QuadratureOp
//===----------------------------------------------------------------------===//

def Quiccir_QuadratureOp : Quiccir_Op<"quadrature", [Pure]> {
    let summary = "Batched backward transform kernel";
    let description = [{
        The `quiccir.quadrature` operation implements a batched quadrature based transform kernel.
        At the moment this is a dense operation, next should be implemented as sparse.

        This operation takes 2 tensors, one representing the operator the second the
        modes/quadrature values, it returns a tensor representing the value at the
        integration points or the modes.

        The shape of the operator and the input tensor need to fit.

        The operation represents oprtr * mods = phys or oprtr * vals = mods.

        Example:

        ```mlir
        %uphys = quiccir.quadrature %oprtr, %umods: tensor<?x?x?x?>, tensor<?x?x?x?> -> tensor<?x?x?x?>
        ```
    }];

    let arguments = (ins AnyTensor:$oprtr, AnyTensor:$umods);
    let results = (outs AnyTensor:$uphys);

    let assemblyFormat = [{
        $oprtr `,` $umods `:`  type($oprtr) `,` type($umods)  `->` type($uphys) attr-dict
    }];

    // Add additional verification logic to the constant operation. Setting this bit
    // to `1` will generate a `::mlir::LogicalResult verify()` declaration on the
    // operation class that is called after ODS constructs have been verified, for
    // example the types of arguments and results. We implement additional verification
    // in the definition of this `verify` method in the C++ source file.
    let hasVerifier = 1;

    // let skipDefaultBuilders = 1;
    // Allow building a QuadratureOp from the two input operands.
    let builders = [
        OpBuilder<(ins "Value":$oprtr, "Value":$umods)>
    ];
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

// def Quiccir_TransposeOp : Quiccir_Op<"transpose", [InferShapedTypeOpAdaptor, Pure]> {
def Quiccir_TransposeOp : Quiccir_Op<"transpose", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "Transpose operator";

    let description = [{
        Permutes the dimensions based on perm.
    }];

    let arguments = (ins
        AnyTensor:$input,
        DenseI64ArrayAttr:$permutation
    );

    let results = (
        outs AnyTensor:$output
    );

    let assemblyFormat = [{
        $input `permutation` `=` $permutation `:` type($input) `->` type($output) attr-dict-with-keyword
    }];

}


//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def Quiccir_AddOp : Quiccir_Op<"add", [Pure, Commutative, Elementwise,
    SameOperandsAndResultShape, SameOperandsAndResultElementType,
    DeclareOpInterfaceMethods<FoldTensorCastIntoConsumerOpInterface>,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "AddOp";
    let description = [{
        The `quiccir.add` operation implements a pointwise addition.
        ?variadic?

        Example:

        ```mlir
        %res = quiccir.add %lhs, %rhs: tensor<?x?x?x?>, tensor<?x?x?x?> -> tensor<?x?x?x?>
        ```
    }];

    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs,
        OptionalAttr<I64Attr>:$implptr);
    let results = (outs AnyTensor:$res);

    let assemblyFormat = [{
        $lhs `,` $rhs `:` type($rhs) `,` type($lhs) `->` type($res) attr-dict-with-keyword
    }];

    // let hasVerifier = 1;
    // let hasCanonicalizeMethod = 1;

    // let skipDefaultBuilders = 1;
    // Allow building a FrPOp from the input operand.
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def Quiccir_SubOp : Quiccir_Op<"sub", [Pure, Elementwise,
    SameOperandsAndResultShape, SameOperandsAndResultElementType,
    DeclareOpInterfaceMethods<FoldTensorCastIntoConsumerOpInterface>,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "SubOp";
    let description = [{
        The `quiccir.sub` operation implements a pointwise subtraction.
        ?variadic?

        Example:

        ```mlir
        %res = quiccir.sub %lhs, %rhs: tensor<?x?x?x?>, tensor<?x?x?x?> -> tensor<?x?x?x?>
        ```
    }];

    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs,
        OptionalAttr<I64Attr>:$implptr);
    let results = (outs AnyTensor:$res);

    let assemblyFormat = [{
        $lhs `,` $rhs `:` type($rhs) `,` type($lhs) `->` type($res) attr-dict-with-keyword
    }];

    // let hasVerifier = 1;
    // let hasCanonicalizeMethod = 1;

    // let skipDefaultBuilders = 1;
    // Allow building a FrPOp from the input operand.
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}


//===----------------------------------------------------------------------===//
// FourierProjectorOp
//===----------------------------------------------------------------------===//

def Quiccir_FrPOp : Quiccir_Op<"fr.prj", [Pure,
    DeclareOpInterfaceMethods<FoldTensorCastIntoConsumerOpInterface>]> {
    let summary = "Fourier Transform modal to physical space";
    let description = [{
        The `quiccir.fr.prj` operation implements a generic Fourier projector.

        This operation takes one tensor representing, the
        modes values, it returns a tensor representing the value at the
        integration points.

        Attributes:
        - implptr, index to retrieve backend class implementation
        - dim, size of the operator that cannot be inferred from the input

        Example:

        ```mlir
        %phys = quiccir.fr.prj<implptr = 0 : i64 : !llvm.ptr, dim = 3 : i64> %mods: tensor<?x?x?x?> -> tensor<?x?x?x?>
        ```
    }];

    let arguments = (ins AnyTensor:$mods
        , OptionalAttr<I64Attr>:$implptr
        , OptionalAttr<I64Attr>:$dim);
    let results = (outs AnyTensor:$phys);

    let assemblyFormat = [{
        $mods `:` type($mods)  `->` type($phys) attr-dict-with-keyword
    }];

    let hasVerifier = 1;
    // let hasCanonicalizeMethod = 1;

    // let skipDefaultBuilders = 1;
    // Allow building a FrPOp from the input operand.
    let builders = [
        OpBuilder<(ins "Value":$mods)>
    ];
}

//===----------------------------------------------------------------------===//
// FourierIntegratorOp
//===----------------------------------------------------------------------===//


def Quiccir_FrIOp : Quiccir_Op<"fr.int", [Pure,
    DeclareOpInterfaceMethods<FoldTensorCastIntoConsumerOpInterface>,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "Fourier Transform from physical to modal space";
    let description = [{
        The `quiccir.fr.int` operation implements a generic Fourier integrator.
        At the momement there is no implemented path for code generation,
        but only lowering to library calls.

        This operation takes one tensor representing, the value at the
        integration points, it returns a tensor representing the
        modes values.

        Attributes:
        - implptr, index to retrieve backend class implementation
        - dim, size of the operator that cannot be inferred from the input

        Example:

        ```mlir
        %mods = quiccir.fr.int<implptr = 0 : i64 : !llvm.ptr, dim = 3 : i64> %phys: tensor<?x?x?x?> -> tensor<?x?x?x?>
        ```
    }];

    let arguments = (ins AnyTensor:$phys
        , OptionalAttr<I64Attr>:$implptr
        , OptionalAttr<I64Attr>:$dim);
    let results = (outs AnyTensor:$mods);

    let assemblyFormat = [{
        $phys `:` type($phys)  `->` type($mods) attr-dict-with-keyword
    }];

    let hasVerifier = 1;

    // let skipDefaultBuilders = 1;
    // Allow building a FrIOp from the input operand.
    let builders = [
        OpBuilder<(ins "Value":$phys)>
    ];

    let extraClassDefinition = inferTransformShapes;
}

//===----------------------------------------------------------------------===//
// AssociateLegendreProjectorOp
//===----------------------------------------------------------------------===//

def Quiccir_AlPOp : Quiccir_Op<"al.prj", [Pure]> {
    let summary = "AssociateLegendre Transform modal to physical space";
    let description = [{
        The `quiccir.al.prj` operation implements a generic AssociateLegendre projector.

        This operation takes one tensor representing, the
        modes values, it returns a tensor representing the value at the
        integration points.

        Attributes:
        - implptr, index to retrieve backend class implementation
        - dim, size of the operator that cannot be inferred from the input

        Example:

        ```mlir
        %phys = quiccir.al.prj<implptr = 0 : i64 : !llvm.ptr, dim = 3 : i64> %mods: tensor<?x?x?x?> -> tensor<?x?x?x?>
        ```
    }];

    let arguments = (ins AnyTensor:$mods
        , OptionalAttr<I64Attr>:$implptr
        , OptionalAttr<I64Attr>:$dim);
    let results = (outs AnyTensor:$phys);

    let assemblyFormat = [{
        $mods `:` type($mods)  `->` type($phys) attr-dict-with-keyword
    }];

    let hasVerifier = 1;

    // let skipDefaultBuilders = 1;
    // Allow building a FrPOp from the input operand.
    let builders = [
        OpBuilder<(ins "Value":$mods)>
    ];
}

//===----------------------------------------------------------------------===//
// AssociateLegendreIntegratorOp
//===----------------------------------------------------------------------===//

def Quiccir_AlIOp : Quiccir_Op<"al.int", [Pure,
    DeclareOpInterfaceMethods<FoldTensorCastIntoConsumerOpInterface>,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "AssociateLegendre Transform from physical to modal space";
    let description = [{
        The `quiccir.al.int` operation implements a generic AssociateLegendre integrator.
        At the momement there is no implemented path for code generation,
        but only lowering to library calls.

        This operation takes one tensor representing, the value at the
        integration points, it returns a tensor representing the
        modes values.

        Attributes:
        - implptr, index to retrieve backend class implementation
        - dim, size of the operator that cannot be inferred from the input

        Example:

        ```mlir
        %mods = quiccir.al.int<implptr = 0 : i64 : !llvm.ptr, dim = 3 : i64> %phys: tensor<?x?x?x?> -> tensor<?x?x?x?>
        ```
    }];

    let arguments = (ins AnyTensor:$phys
        , OptionalAttr<I64Attr>:$implptr
        , OptionalAttr<I64Attr>:$dim);
    let results = (outs AnyTensor:$mods);

    let assemblyFormat = [{
        $phys `:` type($phys)  `->` type($mods) attr-dict-with-keyword
    }];

    let hasVerifier = 1;

    // let skipDefaultBuilders = 1;
    // Allow building a FrIOp from the input operand.
    let builders = [
        OpBuilder<(ins "Value":$phys)>
    ];

    let extraClassDefinition = inferTransformShapes;
}

//===----------------------------------------------------------------------===//
// JonesWorlandProjectorOp
//===----------------------------------------------------------------------===//

def Quiccir_JWPOp : Quiccir_Op<"jw.prj", [Pure]> {
    let summary = "Jone-Worland Transform modal to physical space";
    let description = [{
        The `quiccir.jw.prj` operation implements a generic Jones Worland projector.
        At the momement there is no implemented path for code generation,
        but only lowering to library calls.
        Future codegen path will lower to LinAlg on sparse tensors for
        quadrature ops.

        This operation takes one tensor representing, the
        modes values, it returns a tensor representing the value at the
        integration points.

        Attributes:
        - implptr, index to retrieve backend class implementation
        - dim, size of the operator that cannot be inferred from the input

        Example:

        ```mlir
        %phys = quiccir.jw.prj<implptr = 0 : i64 : !llvm.ptr, dim = 3 : i64> %mods: tensor<?x?x?x?> -> tensor<?x?x?x?>
        ```
    }];

    let arguments = (ins AnyTensor:$mods
        , OptionalAttr<I64Attr>:$implptr
        , OptionalAttr<I64Attr>:$dim);
    let results = (outs AnyTensor:$phys);

    let assemblyFormat = [{
        $mods `:` type($mods)  `->` type($phys) attr-dict-with-keyword
    }];

    let hasVerifier = 1;

    // let skipDefaultBuilders = 1;
    // Allow building a JWPOp from the input operand.
    let builders = [
        OpBuilder<(ins "Value":$mods)>
    ];
}

//===----------------------------------------------------------------------===//
// JonesWorlandIntegratorOp
//===----------------------------------------------------------------------===//

def Quiccir_JWIOp : Quiccir_Op<"jw.int", [Pure]> {
    let summary = "Jone-Worland Transform from physical to modal space";
    let description = [{
        The `quiccir.jw.int` operation implements a generic Jones Worland integrator.
        At the momement there is no implemented path for code generation,
        but only lowering to library calls.
        Future codegen path will lower to LinAlg on sparse tensors for
        quadrature ops.

        This operation takes one tensor representing, the value at the
        integration points, it returns a tensor representing the
        modes values.

        Attributes:
        - implptr, index to retrieve backend class implementation
        - dim, size of the operator that cannot be inferred from the input

        Example:

        ```mlir
        %mods = quiccir.jw.int<implptr = 0 : i64 : !llvm.ptr, dim = 3 : i64> %phys: tensor<?x?x?x?> -> tensor<?x?x?x?>
        ```
    }];

    let arguments = (ins AnyTensor:$phys
        , OptionalAttr<I64Attr>:$implptr
        , OptionalAttr<I64Attr>:$dim);
    let results = (outs AnyTensor:$mods);

    let assemblyFormat = [{
        $phys `:` type($phys)  `->` type($mods) attr-dict-with-keyword
    }];

    let hasVerifier = 1;

    // let skipDefaultBuilders = 1;
    // Allow building a JWIOp from the input operand.
    let builders = [
        OpBuilder<(ins "Value":$phys)>
    ];
}


#endif // QUICCIR_OPS
