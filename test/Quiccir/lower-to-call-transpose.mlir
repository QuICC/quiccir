// RUN: quiccir-opt %s --convert-quiccir-to-call -verify-diagnostics
// RUN: quiccir-opt %s --convert-quiccir-to-call | FileCheck %s

module {
    // materialize to existing buffer
    //
    // CHECK: func.func @entryTransposeBuf(%[[METAARRPTR:.*]]: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %[[THISARRPTR:.*]]: !llvm.ptr<array<1 x ptr>>, %[[V:.*]]: !quiccir.view<16x2x3xf32, "layoutIn">, %[[VTRA:.*]]: !quiccir.view<16x3x2xf32, "layoutOut">) {
    func.func @entryTransposeBuf(%metaArr: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %thisArr: !llvm.ptr<array<1 x ptr>>, %v: !quiccir.view<16x2x3xf32, "layoutIn">, %vtra: !quiccir.view<16x3x2xf32, "layoutOut">) {
        // CHECK: %[[VT:.*]] = builtin.unrealized_conversion_cast %[[V]] : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
        // CHECK: %[[THISARR:.*]] = llvm.load %[[THISARRPTR]] : !llvm.ptr<array<1 x ptr>>
        // CHECK: %[[THIS:.*]] = llvm.extractvalue %[[THISARR]][0] : !llvm.array<1 x ptr>
        // CHECK: call @_ciface_quiccir_transpose_021_f32_layoutOut_f32_layoutIn(%[[THIS]], %[[VTRA]], %[[V]]) : (!llvm.ptr, !quiccir.view<16x3x2xf32, "layoutOut">, !quiccir.view<16x2x3xf32, "layoutIn">) -> ()
        %vt = builtin.unrealized_conversion_cast %v : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
        %tra = quiccir.transpose %vt permutation = [0, 2, 1] : tensor<16x2x3xf32, "layoutIn"> -> tensor<16x3x2xf32, "layoutOut"> attributes{implptr = 0 :i64}
        quiccir.materialize %tra in %vtra : (tensor<16x3x2xf32, "layoutOut">, !quiccir.view<16x3x2xf32, "layoutOut">)
        return
    }

    // alloc new buffer
    //
    // CHECK: func.func @entryTransposeAlloc(%[[METAARRPTR:.*]]: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %[[THISARRPTR:.*]]: !llvm.ptr<array<2 x ptr>>, %[[V:.*]]: !quiccir.view<16x2x3xf32, "layoutIn">) {
    func.func @entryTransposeAlloc(%metaArr: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %thisArr: !llvm.ptr<array<2 x ptr>>, %v: !quiccir.view<16x2x3xf32, "layoutIn">) {
        // CHECK: %[[METAARR:.*]] = llvm.load %[[METAARRPTR]] : !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>
        // CHECK: %[[META0:.*]] = llvm.extractvalue %[[METAARR]][0] : !llvm.array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>
        // CHECK: %[[POINTERS:.*]] = builtin.unrealized_conversion_cast %[[META0]] : !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>> to memref<?xi32>
        // CHECK: %[[META1:.*]] = llvm.extractvalue %[[METAARR]][1] : !llvm.array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>
        // CHECK: %[[INDICES:.*]] = builtin.unrealized_conversion_cast %[[META1]] : !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>> to memref<?xi32>
        // CHECK: %[[LDS:.*]] = llvm.mlir.constant(3 : i64) : i64
        // CHECK: %[[DATA:.*]] = quiccir.alloc_data(%[[POINTERS]], %[[INDICES]]), %[[LDS]] : (memref<?xi32>, memref<?xi32>), i64 -> memref<?xf32> {layout = "layoutOut"}
        // CHECK: %[[VIEW:.*]] = quiccir.assemble(%[[POINTERS]], %[[INDICES]]), %[[DATA]] : (memref<?xi32>, memref<?xi32>), memref<?xf32> -> !quiccir.view<16x3x2xf32, "layoutOut">
        // CHECK: %[[THISARR:.*]] = llvm.load %[[THISARRPTR]] : !llvm.ptr<array<2 x ptr>>
        // CHECK: %[[THIS0:.*]] = llvm.extractvalue %[[THISARR]][0] : !llvm.array<2 x ptr>
        // CHECK: call @_ciface_quiccir_transpose_021_f32_layoutOut_f32_layoutIn(%[[THIS0]], %[[VIEW]], %[[V]]) : (!llvm.ptr, !quiccir.view<16x3x2xf32, "layoutOut">, !quiccir.view<16x2x3xf32, "layoutIn">) -> ()
        %vt = builtin.unrealized_conversion_cast %v : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
        %tra = quiccir.transpose %vt permutation = [0, 2, 1] : tensor<16x2x3xf32, "layoutIn"> -> tensor<16x3x2xf32, "layoutOut"> attributes{implptr = 0 :i64}
        // Note, the following line is neeeded to lowering pass to identify the transpose stage
        // this could be avoid by providing a optional stage attribute to the transpose op
        %prj = quiccir.fr.int %tra : tensor<16x3x2xf32, "layoutOut"> -> tensor<16x3x2xcomplex<f32>, "layoutOutPrj"> attributes{implptr = 1 :i64}
        return
    }

    // materialize to existing buffer, variadic
    //
    // CHECK: func.func @entry2TransposeBuf(%[[METAARRPTR:.*]]: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %[[THISARRPTR:.*]]: !llvm.ptr<array<1 x ptr>>, %[[V0:.*]]: !quiccir.view<16x2x3xf32, "layoutIn">, %[[V1:.*]]: !quiccir.view<16x2x3xf32, "layoutIn">, %[[VTRA0:.*]]: !quiccir.view<16x3x2xf32, "layoutOut">, %[[VTRA1:.*]]: !quiccir.view<16x3x2xf32, "layoutOut">) {
    func.func @entry2TransposeBuf(%metaArr: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %thisArr: !llvm.ptr<array<1 x ptr>>,
        %v0: !quiccir.view<16x2x3xf32, "layoutIn">, %v1: !quiccir.view<16x2x3xf32, "layoutIn">, %vtra0: !quiccir.view<16x3x2xf32, "layoutOut">, %vtra1: !quiccir.view<16x3x2xf32, "layoutOut">) {
        // CHECK: %[[VT0:.*]] = builtin.unrealized_conversion_cast %[[V0]] : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
        // CHECK: %[[VT1:.*]] = builtin.unrealized_conversion_cast %[[V1]] : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
        // CHECK: %[[THISARR:.*]] = llvm.load %[[THISARRPTR]] : !llvm.ptr<array<1 x ptr>>
        // CHECK: %[[THIS:.*]] = llvm.extractvalue %[[THISARR]][0] : !llvm.array<1 x ptr>
        // CHECK: call @_ciface_quiccir_transpose_021_f32_layoutOut_f32_layoutOut_f32_layoutIn_f32_layoutIn(%[[THIS]], %[[VTRA0]], %[[VTRA1]], %[[V0]], %[[V1]]) : (!llvm.ptr, !quiccir.view<16x3x2xf32, "layoutOut">, !quiccir.view<16x3x2xf32, "layoutOut">, !quiccir.view<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">) -> ()
        %vt0 = builtin.unrealized_conversion_cast %v0 : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
        %vt1 = builtin.unrealized_conversion_cast %v1 : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
        %tra:2 = quiccir.transpose %vt0, %vt1 permutation = [0, 2, 1] : tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn"> -> tensor<16x3x2xf32, "layoutOut">, tensor<16x3x2xf32, "layoutOut"> attributes{implptr = 0 :i64}
        quiccir.materialize %tra#0 in %vtra0 : (tensor<16x3x2xf32, "layoutOut">, !quiccir.view<16x3x2xf32, "layoutOut">)
        quiccir.materialize %tra#1 in %vtra1 : (tensor<16x3x2xf32, "layoutOut">, !quiccir.view<16x3x2xf32, "layoutOut">)
        return
    }

    // alloc new buffer, variadic
    //
    // CHECK: func.func @entry2TransposeAlloc(%[[METAARRPTR:.*]]: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %[[THISARRPTR:.*]]: !llvm.ptr<array<2 x ptr>>, %[[V0:.*]]: !quiccir.view<16x2x3xf32, "layoutIn">, %[[V1:.*]]: !quiccir.view<16x2x3xf32, "layoutIn">) {
    func.func @entry2TransposeAlloc(%metaArr: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %thisArr: !llvm.ptr<array<2 x ptr>>,
        // CHECK: %[[VT0:.*]] = builtin.unrealized_conversion_cast %[[V0]] : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
        // CHECK: %[[VT1:.*]] = builtin.unrealized_conversion_cast %[[V1]] : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
        // CHECK: %[[METAARR:.*]] = llvm.load %[[METAARRPTR]] : !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>
        // CHECK: %[[META0:.*]] = llvm.extractvalue %[[METAARR]][0] : !llvm.array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>
        // CHECK: %[[POINTERS:.*]] = builtin.unrealized_conversion_cast %[[META0]] : !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>> to memref<?xi32>
        // CHECK: %[[META1:.*]] = llvm.extractvalue %[[METAARR]][1] : !llvm.array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>
        // CHECK: %[[INDICES:.*]] = builtin.unrealized_conversion_cast %[[META1]] : !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>> to memref<?xi32>
        // CHECK: %[[LDS:.*]] = llvm.mlir.constant(3 : i64) : i64
        // CHECK: %[[DATA0:.*]] = quiccir.alloc_data(%[[POINTERS]], %[[INDICES]]), %[[LDS]] : (memref<?xi32>, memref<?xi32>), i64 -> memref<?xf32> {layout = "layoutOut"}
        // CHECK: %[[VIEW0:.*]] = quiccir.assemble(%[[POINTERS]], %[[INDICES]]), %[[DATA0]] : (memref<?xi32>, memref<?xi32>), memref<?xf32> -> !quiccir.view<16x3x2xf32, "layoutOut">
        // CHECK: %[[METAARR:.*]] = llvm.load %[[METAARRPTR]] : !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>
        // CHECK: %[[META0:.*]] = llvm.extractvalue %[[METAARR]][0] : !llvm.array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>
        // CHECK: %[[POINTERS:.*]] = builtin.unrealized_conversion_cast %[[META0]] : !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>> to memref<?xi32>
        // CHECK: %[[META1:.*]] = llvm.extractvalue %[[METAARR]][1] : !llvm.array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>
        // CHECK: %[[INDICES:.*]] = builtin.unrealized_conversion_cast %[[META1]] : !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>> to memref<?xi32>
        // CHECK: %[[LDS:.*]] = llvm.mlir.constant(3 : i64) : i64
        // CHECK: %[[DATA1:.*]] = quiccir.alloc_data(%[[POINTERS]], %[[INDICES]]), %[[LDS]] : (memref<?xi32>, memref<?xi32>), i64 -> memref<?xf32> {layout = "layoutOut"}
        // CHECK: %[[VIEW1:.*]] = quiccir.assemble(%[[POINTERS]], %[[INDICES]]), %[[DATA1]] : (memref<?xi32>, memref<?xi32>), memref<?xf32> -> !quiccir.view<16x3x2xf32, "layoutOut">
        // CHECK: %[[THISARR:.*]] = llvm.load %[[THISARRPTR]] : !llvm.ptr<array<2 x ptr>>
        // CHECK: %[[THIS0:.*]] = llvm.extractvalue %[[THISARR]][0] : !llvm.array<2 x ptr>
        // CHECK: call @_ciface_quiccir_transpose_021_f32_layoutOut_f32_layoutOut_f32_layoutIn_f32_layoutIn(%[[THIS0]], %[[VIEW0]], %[[VIEW1]], %[[V0]], %[[V1]]) : (!llvm.ptr, !quiccir.view<16x3x2xf32, "layoutOut">, !quiccir.view<16x3x2xf32, "layoutOut">, !quiccir.view<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">) -> ()
        %v0: !quiccir.view<16x2x3xf32, "layoutIn">, %v1: !quiccir.view<16x2x3xf32, "layoutIn">) {
        %vt0 = builtin.unrealized_conversion_cast %v0 : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
        %vt1 = builtin.unrealized_conversion_cast %v1 : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
        %tra:2 = quiccir.transpose %vt0, %vt1 permutation = [0, 2, 1] : tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn"> -> tensor<16x3x2xf32, "layoutOut">, tensor<16x3x2xf32, "layoutOut"> attributes{implptr = 0 :i64}
        %prj0 = quiccir.fr.int %tra#0 : tensor<16x3x2xf32, "layoutOut"> -> tensor<16x3x2xcomplex<f32>, "layoutOutPrj"> attributes{kind = "P", implptr = 1 :i64}
        %prj1 = quiccir.fr.int %tra#1 : tensor<16x3x2xf32, "layoutOut"> -> tensor<16x3x2xcomplex<f32>, "layoutOutPrj"> attributes{kind = "P", implptr = 1 :i64}
        return
    }
}
