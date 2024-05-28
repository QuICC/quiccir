// RUN: quiccir-opt %s --lower-quiccir-alloc | FileCheck %s

module {
  func.func private @user(!quiccir.view<16x3x3xf32, "layoutUval">)
  func.func @entry1(%arg1: !quiccir.view<16x2x3xf32, "layoutUmod">) {
    // CHECK: %[[ST0:.*]] = llvm.mlir.undef : !llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f32>, i32)>
    // CHECK: %[[DIM0:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: %[[DIM1:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK: %[[DIM2:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK: %[[ST1:.*]] = llvm.insertvalue %[[DIM0]], %[[ST0]][0, 2] : !llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f32>, i32)>
    // CHECK: %[[ST2:.*]] = llvm.insertvalue %[[DIM1]], %[[ST1]][0, 0] : !llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f32>, i32)>
    // CHECK: %{{.*}} = llvm.insertvalue %[[DIM2]], %[[ST2]][0, 1] : !llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f32>, i32)>
    // CHECK: %{{.*}} = llvm.mlir.constant(1 : index) : i64
    // CHECK: %{{.*}} = llvm.alloca %{{.*}} x !llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f32>, i32)> : (i64) -> !llvm.ptr<struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f32>, i32)>>
    // CHECK: lvm.store %{{.*}}, %{{.*}} : !llvm.ptr<struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f32>, i32)>>
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !llvm.ptr<struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f32>, i32)>> to !quiccir.view<16x3x3xf32, "layoutUval">
    // CHECK: call @_ciface_quiccir_alloc_jw_prj_f32_layoutUval_f32_layoutUmod(%{{.*}}, %{{.*}}) : (!quiccir.view<16x3x3xf32, "layoutUval">, !quiccir.view<16x2x3xf32, "layoutUmod">) -> ()
    %0 = quiccir.alloc(%arg1) : !quiccir.view<16x2x3xf32, "layoutUmod"> -> !quiccir.view<16x3x3xf32, "layoutUval"> {producer = "quiccir.jw.prj"}
    call @user(%0) : (!quiccir.view<16x3x3xf32, "layoutUval">) -> ()
    return
  }

  func.func @entry2(%arg1: !quiccir.view<16x2x3xf32, "layoutUmod">) {
    // CHECK: %[[V:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !quiccir.view<16x2x3xf32, "layoutUmod"> to !quiccir.view<?x?x?xf32, "layoutUmod"
    // CHECK: call @_ciface_quiccir_dealloc_f32_layoutUmod(%[[V]]) : (!quiccir.view<?x?x?xf32, "layoutUmod">) -> ()
    quiccir.dealloc(%arg1) : !quiccir.view<16x2x3xf32, "layoutUmod">
    return
  }
}