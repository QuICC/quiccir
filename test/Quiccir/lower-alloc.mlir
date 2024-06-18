// RUN: quiccir-opt %s --lower-quiccir-alloc | FileCheck %s

module {

  func.func @allocData(%ptr: memref<?xi32>, %idx: memref<?xi32>) {
    %lds = llvm.mlir.constant(3 : i64) : i64
    // CHECK: %[[ST:.*]] = llvm.alloca %1 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>
    // CHECK: %[[MR:.*]] = builtin.unrealized_conversion_cast %[[ST]] : !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>> to memref<?xcomplex<f32>>
    // CHECK: call @_ciface_quiccir_alloc_data_complexf32_i32_i32_layoutNew(%[[MR]], %{{.*}}, %{{.*}}, %{{.*}}) : (memref<?xcomplex<f32>>, memref<?xi32>, memref<?xi32>, i64) -> ()
    %data = quiccir.alloc_data(%ptr, %idx), %lds : (memref<?xi32>, memref<?xi32>), i64 -> memref<?xcomplex<f32>> {layout = "layoutNew"}
    return
  }

  func.func @dealloc(%ptr: memref<?xi32>, %idx: memref<?xi32>, %data: memref<?xf32>) {
    %view = quiccir.assemble(%ptr, %idx), %data : (memref<?xi32>, memref<?xi32>), memref<?xf32> -> !quiccir.view<16x3x3xf32, "layoutNew">
    // CHECK: %[[V:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !quiccir.view<16x3x3xf32, "layoutNew"> to !quiccir.view<?x?x?xf32, "layoutNew">
    // CHECK: call @_ciface_quiccir_dealloc_f32_layoutNew(%[[V]]) : (!quiccir.view<?x?x?xf32, "layoutNew">) -> ()
    quiccir.dealloc(%view) : !quiccir.view<16x3x3xf32, "layoutNew">
    return
  }
}