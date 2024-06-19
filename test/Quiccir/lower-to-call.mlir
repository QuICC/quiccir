// RUN: quiccir-opt %s --convert-quiccir-to-call -verify-diagnostics
// RUN: quiccir-opt %s --convert-quiccir-to-call | FileCheck %s

module {
    // materialize to existing buffer
    func.func @entryJwPrjBuf(%metaArr: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %thisArr: !llvm.ptr<array<1 x ptr>>, %vuval: !quiccir.view<16x3x3xcomplex<f32>, "layoutUval">, %vumod: !quiccir.view<16x2x3xcomplex<f32>, "layoutUmod">) {
    %umod = builtin.unrealized_conversion_cast %vumod : !quiccir.view<16x2x3xcomplex<f32>, "layoutUmod"> to tensor<16x2x3xcomplex<f32>, "layoutUmod">
    // CHECK: %[[ARR:.*]] = llvm.load %{{.*}} : !llvm.ptr<array<1 x ptr>>
    // CHECK: %[[THIS:.*]] = llvm.extractvalue %[[ARR]][0] : !llvm.array<1 x ptr>
    // CHECK: call @_ciface_quiccir_jw_prj_complexf32_layoutUval_complexf32_layoutUmod(%[[THIS]], %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<16x3x3xcomplex<f32>, "layoutUval">, !quiccir.view<16x2x3xcomplex<f32>, "layoutUmod">) -> ()
    %ret = quiccir.jw.prj %umod : tensor<16x2x3xcomplex<f32>, "layoutUmod"> -> tensor<16x3x3xcomplex<f32>, "layoutUval"> attributes{implptr = 0 :i64}
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !quiccir.view<16x3x3xcomplex<f32>, "layoutUval"> to tensor<16x3x3xcomplex<f32>, "layoutUval">
    quiccir.materialize %ret in %vuval : (tensor<16x3x3xcomplex<f32>, "layoutUval">, !quiccir.view<16x3x3xcomplex<f32>, "layoutUval">)
    return
    }

    // create new buffer reusing stage meta data
    func.func @entryJwPrjAlloc(%metaArr: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %thisArr: !llvm.ptr<array<2 x ptr>>, %vumod: !quiccir.view<16x2x3xcomplex<f32>, "layoutUmod">) {
    %umod = builtin.unrealized_conversion_cast %vumod : !quiccir.view<16x2x3xcomplex<f32>, "layoutUmod"> to tensor<16x2x3xcomplex<f32>, "layoutUmod">
    // CHECK: %[[PTR:.*]] = quiccir.pointers %{{.*}} : !quiccir.view<16x2x3xcomplex<f32>, "layoutUmod"> -> memref<?xi32>
    // CHECK: %[[IDX:.*]] = quiccir.indices %{{.*}} : !quiccir.view<16x2x3xcomplex<f32>, "layoutUmod"> -> memref<?xi32>
    // CHECK: %[[LDS:.*]] = llvm.mlir.constant(3 : i64) : i64
    // CHECK: %[[DATA:.*]] = quiccir.alloc_data(%[[PTR]], %[[IDX]]), %[[LDS]] : (memref<?xi32>, memref<?xi32>), i64 -> memref<?xcomplex<f32>> {layout = "layoutUval"}
    // CHECK: %[[VIEW:.*]] = quiccir.assemble(%[[PTR]], %[[IDX]]), %[[DATA]] : (memref<?xi32>, memref<?xi32>), memref<?xcomplex<f32>> -> !quiccir.view<16x3x3xcomplex<f32>, "layoutUval">
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<array<2 x ptr>>
    // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x ptr>
    // CHECK: call @_ciface_quiccir_jw_prj_complexf32_layoutUval_complexf32_layoutUmod(%{{.*}}, %[[VIEW]], %{{.*}}) : (!llvm.ptr, !quiccir.view<16x3x3xcomplex<f32>, "layoutUval">, !quiccir.view<16x2x3xcomplex<f32>, "layoutUmod">) -> ()
    %ret = quiccir.jw.prj %umod : tensor<16x2x3xcomplex<f32>, "layoutUmod"> -> tensor<16x3x3xcomplex<f32>, "layoutUval"> attributes{implptr = 1 :i64}
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !quiccir.view<16x3x3xcomplex<f32>, "layoutUval"> to tensor<16x3x3xcomplex<f32>, "layoutUval">
    return
    }

    // materialize to existing buffer
    func.func @entryTransposeBuf(%metaArr: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %thisArr: !llvm.ptr<array<1 x ptr>>, %v: !quiccir.view<16x2x3xf32, "layoutIn">, %vtra: !quiccir.view<16x3x2xf32, "layoutOut">) {
    %vt = builtin.unrealized_conversion_cast %v : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<array<1 x ptr>>
    // CHECK: %{{.*}} = llvm.extractvalue %{{.*}}[0] : !llvm.array<1 x ptr>
    // CHECK: call @_ciface_quiccir_transpose_021_f32_layoutOut_f32_layoutIn(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<16x3x2xf32, "layoutOut">, !quiccir.view<16x2x3xf32, "layoutIn">) -> ()
    %tra = quiccir.transpose %vt permutation = [0, 2, 1] : tensor<16x2x3xf32, "layoutIn"> -> tensor<16x3x2xf32, "layoutOut"> attributes{implptr = 0 :i64}
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !quiccir.view<16x3x2xf32, "layoutOut"> to tensor<16x3x2xf32, "layoutOut">
    quiccir.materialize %tra in %vtra : (tensor<16x3x2xf32, "layoutOut">, !quiccir.view<16x3x2xf32, "layoutOut">)
    return
    }
}
