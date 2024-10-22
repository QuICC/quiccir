// RUN: quiccir-opt %s --convert-quiccir-to-call -verify-diagnostics
// RUN: quiccir-opt %s --convert-quiccir-to-call | FileCheck %s

module {
    // create new buffer
    func.func @entrySubAlloc(%metaArr: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %thisArr: !llvm.ptr<array<1 x ptr>>, %a: !quiccir.view<16x2x3xf32, "layoutIn">, %b: !quiccir.view<16x2x3xf32, "layoutIn">) {
    %at = builtin.unrealized_conversion_cast %a : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    %bt = builtin.unrealized_conversion_cast %b : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    // CHECK: %[[PTR:.*]] = quiccir.pointers %{{.*}} : !quiccir.view<16x2x3xf32, "layoutIn"> -> memref<?xi32>
    // CHECK: %[[IDX:.*]] = quiccir.indices %{{.*}} : !quiccir.view<16x2x3xf32, "layoutIn"> -> memref<?xi32>
    // CHECK: %[[LDS:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[DATA:.*]] = quiccir.alloc_data(%[[PTR]], %[[IDX]]), %[[LDS]] : (memref<?xi32>, memref<?xi32>), i64 -> memref<?xf32> {layout = "layoutIn"}
    // CHECK: %[[VIEW:.*]] = quiccir.assemble(%[[PTR]], %[[IDX]]), %[[DATA]] : (memref<?xi32>, memref<?xi32>), memref<?xf32> -> !quiccir.view<16x2x3xf32, "layoutIn">
    // CHECK: %[[ARR:.*]] = llvm.load %{{.*}} : !llvm.ptr<array<1 x ptr>>
    // CHECK: %[[THIS:.*]] = llvm.extractvalue %[[ARR]][0] : !llvm.array<1 x ptr>
    // CHECK: %[[VIEWNODIM:.*]] = builtin.unrealized_conversion_cast %[[VIEW]] : !quiccir.view<16x2x3xf32, "layoutIn"> to !quiccir.view<?x?x?xf32, "layoutIn">
    // CHECK: call @_ciface_quiccir_sub_f32_layoutIn_f32_layoutIn_f32_layoutIn(%[[THIS]], %[[VIEWNODIM]], %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<?x?x?xf32, "layoutIn">, !quiccir.view<?x?x?xf32, "layoutIn">, !quiccir.view<?x?x?xf32, "layoutIn">) -> ()
    %sub = quiccir.sub %at, %bt : tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn"> -> tensor<16x2x3xf32, "layoutIn"> attributes{implptr = 0 :i64}
    return
    }

    // materialize to existing buffer
    func.func @entrySubBuf(%metaArr: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %thisArr: !llvm.ptr<array<1 x ptr>>,  %c: !quiccir.view<16x2x3xf32, "layoutIn">, %a: !quiccir.view<16x2x3xf32, "layoutIn">, %b: !quiccir.view<16x2x3xf32, "layoutIn">) {
    %at = builtin.unrealized_conversion_cast %a : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    %bt = builtin.unrealized_conversion_cast %b : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    // CHECK: %[[ARR:.*]] = llvm.load %{{.*}} : !llvm.ptr<array<1 x ptr>>
    // CHECK: %[[THIS:.*]] = llvm.extractvalue %[[ARR]][0] : !llvm.array<1 x ptr>
    // CHECK: call @_ciface_quiccir_sub_f32_layoutIn_f32_layoutIn_f32_layoutIn(%[[THIS]], %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<?x?x?xf32, "layoutIn">, !quiccir.view<?x?x?xf32, "layoutIn">, !quiccir.view<?x?x?xf32, "layoutIn">) -> ()
    %sub = quiccir.sub %at, %bt : tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn"> -> tensor<16x2x3xf32, "layoutIn"> attributes{implptr = 0 :i64}
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    quiccir.materialize %sub in %c : (tensor<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">)
    return
    }

    // materialize to existing buffer and multiple use
    // CHECK: func.func @entrySubBufUse(%arg0: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %{{.*}}: !llvm.ptr<array<1 x ptr>>, %[[A3:.*]]: !quiccir.view<16x2x3xf32, "layoutIn">, %{{.*}}: !quiccir.view<16x2x3xf32, "layoutIn">, %{{.*}}: !quiccir.view<16x2x3xf32, "layoutIn">) {
    func.func @entrySubBufUse(%metaArr: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %thisArr: !llvm.ptr<array<1 x ptr>>, %c: !quiccir.view<16x2x3xf32, "layoutIn">, %a: !quiccir.view<16x2x3xf32, "layoutIn">, %b: !quiccir.view<16x2x3xf32, "layoutIn">) {
    %at = builtin.unrealized_conversion_cast %a : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    %bt = builtin.unrealized_conversion_cast %b : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    // CHECK: %[[ARR:.*]] = llvm.load %{{.*}} : !llvm.ptr<array<1 x ptr>>
    // CHECK: %[[THIS:.*]] = llvm.extractvalue %[[ARR]][0] : !llvm.array<1 x ptr>
    // CHECK: %[[A3NODIM:.*]] = builtin.unrealized_conversion_cast %[[A3]] : !quiccir.view<16x2x3xf32, "layoutIn"> to !quiccir.view<?x?x?xf32, "layoutIn">
    // CHECK: call @_ciface_quiccir_sub_f32_layoutIn_f32_layoutIn_f32_layoutIn(%[[THIS]], %[[A3NODIM]], %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<?x?x?xf32, "layoutIn">, !quiccir.view<?x?x?xf32, "layoutIn">, !quiccir.view<?x?x?xf32, "layoutIn">) -> ()
    %sub0 = quiccir.sub %at, %bt : tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn"> -> tensor<16x2x3xf32, "layoutIn"> attributes{implptr = 0 :i64}
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    quiccir.materialize %sub0 in %c : (tensor<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">)
    // CHECK: %[[PTR:.*]] = quiccir.pointers %{{.*}} : !quiccir.view<16x2x3xf32, "layoutIn"> -> memref<?xi32>
    // CHECK: %[[IDX:.*]] = quiccir.indices %{{.*}} : !quiccir.view<16x2x3xf32, "layoutIn"> -> memref<?xi32>
    // CHECK: %[[LDS:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[DATA:.*]] = quiccir.alloc_data(%[[PTR]], %[[IDX]]), %[[LDS]] : (memref<?xi32>, memref<?xi32>), i64 -> memref<?xf32> {layout = "layoutIn"}
    // CHECK: %[[VIEW:.*]] = quiccir.assemble(%[[PTR]], %[[IDX]]), %[[DATA]] : (memref<?xi32>, memref<?xi32>), memref<?xf32> -> !quiccir.view<16x2x3xf32, "layoutIn">
    // CHECK: %[[VIEWNODIM:.*]] = builtin.unrealized_conversion_cast %[[VIEW]] : !quiccir.view<16x2x3xf32, "layoutIn"> to !quiccir.view<?x?x?xf32, "layoutIn">
    // CHECK: %[[A3NODIM0:.*]] = builtin.unrealized_conversion_cast %[[A3]] : !quiccir.view<16x2x3xf32, "layoutIn"> to !quiccir.view<?x?x?xf32, "layoutIn">
    // CHECK: call @_ciface_quiccir_sub_f32_layoutIn_f32_layoutIn_f32_layoutIn(%{{.*}}, %[[VIEWNODIM]], %{{.*}}, %[[A3NODIM0]]) : (!llvm.ptr, !quiccir.view<?x?x?xf32, "layoutIn">, !quiccir.view<?x?x?xf32, "layoutIn">, !quiccir.view<?x?x?xf32, "layoutIn">) -> ()
    %sub1 = quiccir.sub %at, %sub0 : tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn"> -> tensor<16x2x3xf32, "layoutIn"> attributes{implptr = 0 :i64}
    return
    }

    // create new buffer
    func.func @entryAddAlloc(%metaArr: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %thisArr: !llvm.ptr<array<1 x ptr>>, %a: !quiccir.view<16x2x3xf32, "layoutIn">, %b: !quiccir.view<16x2x3xf32, "layoutIn">) {
    %at = builtin.unrealized_conversion_cast %a : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    %bt = builtin.unrealized_conversion_cast %b : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    // CHECK: %[[PTR:.*]] = quiccir.pointers %{{.*}} : !quiccir.view<16x2x3xf32, "layoutIn"> -> memref<?xi32>
    // CHECK: %[[IDX:.*]] = quiccir.indices %{{.*}} : !quiccir.view<16x2x3xf32, "layoutIn"> -> memref<?xi32>
    // CHECK: %[[LDS:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[DATA:.*]] = quiccir.alloc_data(%[[PTR]], %[[IDX]]), %[[LDS]] : (memref<?xi32>, memref<?xi32>), i64 -> memref<?xf32> {layout = "layoutIn"}
    // CHECK: %[[VIEW:.*]] = quiccir.assemble(%[[PTR]], %[[IDX]]), %[[DATA]] : (memref<?xi32>, memref<?xi32>), memref<?xf32> -> !quiccir.view<16x2x3xf32, "layoutIn">
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<array<1 x ptr>>
    // CHECK: %[[THIS:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<1 x ptr>
    // CHECK: %[[VIEWNODIM:.*]] = builtin.unrealized_conversion_cast %[[VIEW]] : !quiccir.view<16x2x3xf32, "layoutIn"> to !quiccir.view<?x?x?xf32, "layoutIn">
    // CHECK: call @_ciface_quiccir_add_f32_layoutIn_f32_layoutIn_f32_layoutIn(%[[THIS]], %[[VIEWNODIM]], %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<?x?x?xf32, "layoutIn">, !quiccir.view<?x?x?xf32, "layoutIn">, !quiccir.view<?x?x?xf32, "layoutIn">) -> ()
    %tra = quiccir.add %at, %bt : tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn"> -> tensor<16x2x3xf32, "layoutIn"> attributes{implptr = 0 :i64}
    return
    }
}