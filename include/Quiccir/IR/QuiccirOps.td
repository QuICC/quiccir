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
include "Quiccir/Interfaces/KindOpInterface.td"
include "Quiccir/Interfaces/TransformOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


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
// AllocDataOp
//===----------------------------------------------------------------------===//

def Quiccir_AllocDataOp : Quiccir_Op<"alloc_data", []> {
    let summary = "Allocate memref data for a view";
    let description = [{
        The `quiccir.alloc_data` operation allocate a 1D memref that is
        the buffer for the compressed data of a view.

        In order to allocate the right amount of space for the data
        we need to know view layout, column height (possibly padded for FFT)
        and the pointers/indexes metadata.

        Example:

        ```mlir
        %0 = quiccir.alloc_data(%ptr, %idx), %lds : (memref<xi32>, memref<xi32>), i64 ->
            memref<?xf64>  {layout="layout"}
        ```

        For codegen we should lower this to a memref alloc of appropriate size

    }];

    let arguments = (ins AnyMemRef:$ptr, AnyMemRef:$idx, I64:$lds, StrAttr:$layout);
    let results = (outs AnyMemRef:$data);

    let assemblyFormat = [{
        `(` $ptr `,` $idx `)` `,` $lds `:` `(`type($ptr) `,` type($idx) `)` `,` type($lds) `->` type($data) attr-dict
    }];

    // let skipDefaultBuilders = 1;
    // let builders = [
    //     OpBuilder<(ins "MemRef":$ptr, "MemRef":$idx, "StringAttr":$layout), [{
    //         return build($_builder, $_state, ptr, idx, layout);
    //     }]>
    // ];

    // let hasVerifier = 1;
}

// //===----------------------------------------------------------------------===//
// // DeallocDataOp
// //===----------------------------------------------------------------------===//

// def Quiccir_DeallocDataOp : Quiccir_Op<"dealloc_data", []> {
//     let summary = "Deallocate view data buffer";
//     let description = [{
//         The `quiccir.dealloc_data` operation deallocate a view buffer.

//         Example:

//         ```mlir
//         quiccir.dealloc_data(%arg) : !quiccir.view<?x?x?xf64, "layout">
//         ```
//     }];

//     let arguments = (ins AnyView:$view);

//     let assemblyFormat = [{
//         `(` $view `)` `:` qualified(type($view)) attr-dict
//     }];

//     // let skipDefaultBuilders = 1;
//     // let builders = [
//     //     OpBuilder<(ins "Value":$view), [{
//     //         return build($_builder, $_state, view);
//     //     }]>
//     // ];

//     let hasVerifier = 1;
// }

//===----------------------------------------------------------------------===//
// AssembleOp
//===----------------------------------------------------------------------===//

def Quiccir_AssembleOp : Quiccir_Op<"assemble", []> {
    let summary = "Assemble view from meta and data MemRefs";
    let description = [{
        The `quiccir.assemble` create a view from its data and meta data.

        Example:

        ```mlir
        %0 = quiccir.assemble(%ptr, %idx), %data : (memref<xi32>, memref<xi32>), memref<?xf64> ->
            !quiccir.view<?x?x?xf64, "layout">
        ```

    }];

    let arguments = (ins AnyMemRef:$ptr, AnyMemRef:$idx, AnyMemRef:$data);
    let results = (outs AnyView:$view);

    let assemblyFormat = [{
        `(` $ptr `,` $idx `)` `,` $data `:` `(`type($ptr) `,` type($idx) `)` `,` type($data) `->` qualified(type($view)) attr-dict
    }];

    // let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PointersOp
//===----------------------------------------------------------------------===//

def Quiccir_PointersOp : Quiccir_Op<"pointers", []> {
    let summary = "Pointers view from meta and data MemRefs";
    let description = [{
        The `quiccir.pointers` extract pointers meta data from a view.

        Example:

        ```mlir
        %ptr = quiccir.pointers %view : !quiccir.view<?x?x?xf64, "layout"> -> memref<?xi32>
        ```
    }];

    let arguments = (ins AnyView:$view);
    let results = (outs AnyMemRef:$ptr);

    let assemblyFormat = [{
        $view `:` qualified(type($view)) `->` type($ptr) attr-dict
    }];

    // let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IndicesOp
//===----------------------------------------------------------------------===//

def Quiccir_IndicesOp : Quiccir_Op<"indices", []> {
    let summary = "Indices view from meta and data MemRefs";
    let description = [{
        The `quiccir.indices` extract indices meta data from a view.

        Example:

        ```mlir
        %idx = quiccir.indices %view : !quiccir.view<?x?x?xf64, "layout"> -> memref<?xi32>
        ```
    }];

    let arguments = (ins AnyView:$view);
    let results = (outs AnyMemRef:$idx);

    let assemblyFormat = [{
        $view `:` qualified(type($view)) `->` type($idx) attr-dict
    }];

    // let hasVerifier = 1;
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
        Permutes the dimensions based on permutation.

        Example:

        ```mlir
        %out = quiccir.transpose %in permutation [0, 2, 1]: tensor<1x2x3xf32> -> tensor<1x3x2xf32>
        ```
    }];

    let arguments = (ins
        AnyTensor:$input,
        DenseI64ArrayAttr:$permutation,
        OptionalAttr<I64Attr>:$implptr
    );

    let results = (
        outs AnyTensor:$output
    );

    let assemblyFormat = [{
        $input `permutation` `=` $permutation `:` type($input) `->` type($output) attr-dict-with-keyword
    }];

    let hasVerifier = 1;
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
// CrossOp
//===----------------------------------------------------------------------===//

def Quiccir_CrossOp : Quiccir_Op<"cross", [Pure,
    SameOperandsShape,
    SameOperandsElementType,
    SameOperandsAndResultShape,
    SameOperandsAndResultElementType,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<KindOpInterface>]> {
    let summary = "Cross product of 2 vectors";
    let description = [{
        The `quiccir.cross` operation implements a cross product of 2 vectors
        in component wise form, each component is a tensor.

        This operation takes 6 tensors representing the vectors,
        it returns 3 tensors the cross product.

        Attributes:
        - implptr, index to retrieve backend class implementation
        - kind, described the constant scaling factor

        (The scaling factor should be exposed in Mlir for code generation)

        Example:

        ```mlir
        %C0, %C1, %C2 = quiccir.cross (%V0, %V1, %V2), (%U0, %U1, %U2) :
        (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>), (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) ->
        (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>)
        attributes{implptr = 0, kind = "inertia"}
        ```
    }];

    let arguments = (ins AnyTensor:$V0, AnyTensor:$V1, AnyTensor:$V2
        , AnyTensor:$U0, AnyTensor:$U1, AnyTensor:$U2
        , OptionalAttr<I64Attr>:$implptr
        , OptionalAttr<StrAttr>:$kind);
    let results = (outs AnyTensor:$C0, AnyTensor:$C1, AnyTensor:$C2);

    let assemblyFormat = [{
        `(` $V0 `,` $V1 `,` $V2 `)` `,` `(` $U0 `,` $U1 `,` $U2 `)` `:` `(` type($V0) `,` type($V1) `,` type($V2) `)`
        `,` `(` type($U0) `,` type($U1) `,` type($U2) `)`  `->` `(` type($C0) `,` type($C1) `,` type($C2) `)`
        attr-dict-with-keyword
    }];

    // let hasVerifier = 1;
    // let hasCanonicalizeMethod = 1;

    // let skipDefaultBuilders = 1;
    // Allow building a CrossOp from the input operands.
    let builders = [
        OpBuilder<(ins "Value":$V0, "Value":$V1, "Value":$V2, "Value":$U0, "Value":$U1, "Value":$U2)>
    ];
}


//===----------------------------------------------------------------------===//
// FourierProjectorOp
//===----------------------------------------------------------------------===//

def Quiccir_FrPOp : Quiccir_Op<"fr.prj", [Pure,
    DeclareOpInterfaceMethods<TransformOpInterface>]> {
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
        , OptionalAttr<I64Attr>:$dim
        , OptionalAttr<StrAttr>:$kind);
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

    let extraClassDefinition = inferTransformShapes;
}

//===----------------------------------------------------------------------===//
// FourierIntegratorOp
//===----------------------------------------------------------------------===//


def Quiccir_FrIOp : Quiccir_Op<"fr.int", [Pure,
    DeclareOpInterfaceMethods<TransformOpInterface>]> {
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
        , OptionalAttr<I64Attr>:$dim
        , OptionalAttr<StrAttr>:$kind);
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

def Quiccir_AlPOp : Quiccir_Op<"al.prj", [Pure,
    DeclareOpInterfaceMethods<TransformOpInterface>]> {
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
        , OptionalAttr<I64Attr>:$dim
        , OptionalAttr<StrAttr>:$kind);
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

    let extraClassDefinition = inferTransformShapes;
}

//===----------------------------------------------------------------------===//
// AssociateLegendreIntegratorOp
//===----------------------------------------------------------------------===//

def Quiccir_AlIOp : Quiccir_Op<"al.int", [Pure,
    DeclareOpInterfaceMethods<TransformOpInterface>]> {
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
        , OptionalAttr<I64Attr>:$dim
        , OptionalAttr<StrAttr>:$kind);
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

def Quiccir_JWPOp : Quiccir_Op<"jw.prj", [Pure,
    DeclareOpInterfaceMethods<TransformOpInterface>]> {
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
        , OptionalAttr<I64Attr>:$dim
        , OptionalAttr<StrAttr>:$kind);
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

    let extraClassDefinition = inferTransformShapes;
}

//===----------------------------------------------------------------------===//
// JonesWorlandIntegratorOp
//===----------------------------------------------------------------------===//

def Quiccir_JWIOp : Quiccir_Op<"jw.int", [Pure,
    DeclareOpInterfaceMethods<TransformOpInterface>]> {
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
        , OptionalAttr<I64Attr>:$dim
        , OptionalAttr<StrAttr>:$kind);
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

    let extraClassDefinition = inferTransformShapes;
}


#endif // QUICCIR_OPS
