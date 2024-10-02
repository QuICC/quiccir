// RUN: quiccir-opt %s --convert-quiccir-to-call -verify-diagnostics
// RUN: quiccir-opt %s --convert-quiccir-to-call | FileCheck %s

module {
    // create new buffer
    func.func @entryCrossAlloc(%metaArr: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %thisArr: !llvm.ptr<array<1 x ptr>>,
    %u0: !quiccir.view<16x2x3xf32, "layoutIn">,
    %u1: !quiccir.view<16x2x3xf32, "layoutIn">,
    %u2: !quiccir.view<16x2x3xf32, "layoutIn">,
    %v0: !quiccir.view<16x2x3xf32, "layoutIn">,
    %v1: !quiccir.view<16x2x3xf32, "layoutIn">,
    %v2: !quiccir.view<16x2x3xf32, "layoutIn">) {
    %u0t = builtin.unrealized_conversion_cast %u0 : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    %u1t = builtin.unrealized_conversion_cast %u1 : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    %u2t = builtin.unrealized_conversion_cast %u2 : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    %v0t = builtin.unrealized_conversion_cast %v0 : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    %v1t = builtin.unrealized_conversion_cast %v1 : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    %v2t = builtin.unrealized_conversion_cast %v2 : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    // CHECK: %[[PTR:.*]] = quiccir.pointers %{{.*}} : !quiccir.view<16x2x3xf32, "layoutIn"> -> memref<?xi32>
    // CHECK: %[[IDX:.*]] = quiccir.indices %{{.*}} : !quiccir.view<16x2x3xf32, "layoutIn"> -> memref<?xi32>
    // CHECK: %[[LDS:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[DATA:.*]] = quiccir.alloc_data(%[[PTR]], %[[IDX]]), %[[LDS]] : (memref<?xi32>, memref<?xi32>), i64 -> memref<?xf32> {layout = "layoutIn"}
    // CHECK: %[[C0:.*]] = quiccir.assemble(%[[PTR]], %[[IDX]]), %[[DATA]] : (memref<?xi32>, memref<?xi32>), memref<?xf32> -> !quiccir.view<16x2x3xf32, "layoutIn">
    // CHECK: %[[C1:.*]] = quiccir.assemble(%{{.*}}, %{{.*}}), %{{.*}} : (memref<?xi32>, memref<?xi32>), memref<?xf32> -> !quiccir.view<16x2x3xf32, "layoutIn">
    // CHECK: %[[C2:.*]] = quiccir.assemble(%{{.*}}, %{{.*}}), %{{.*}} : (memref<?xi32>, memref<?xi32>), memref<?xf32> -> !quiccir.view<16x2x3xf32, "layoutIn">
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<array<1 x ptr>>
    // CHECK: %[[THIS:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<1 x ptr>
    // CHECK: call @_ciface_quiccir_cross_f32_layoutIn_f32_layoutIn_f32_layoutIn_f32_layoutIn_f32_layoutIn_f32_layoutIn_f32_layoutIn_f32_layoutIn_f32_layoutIn(%[[THIS]], %[[C0]], %[[C1]], %[[C2]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">) -> ()
    %C:3 = quiccir.cross (%u0t, %u1t, %u2t), (%v0t, %v1t, %v2t) : (tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn"> ),
        (tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn"> ) -> (tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn"> ) attributes{implptr = 0 :i64}
    return
    }
}
