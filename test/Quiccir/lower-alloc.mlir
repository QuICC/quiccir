// RUN: quiccir-opt %s --lower-quiccir-alloc | FileCheck %s

module {

  func.func @entry2(%arg1: !quiccir.view<16x2x3xf32, "layoutUmod">) {
    // CHECK: %[[V:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !quiccir.view<16x2x3xf32, "layoutUmod"> to !quiccir.view<?x?x?xf32, "layoutUmod"
    // CHECK: call @_ciface_quiccir_dealloc_f32_layoutUmod(%[[V]]) : (!quiccir.view<?x?x?xf32, "layoutUmod">) -> ()
    quiccir.dealloc(%arg1) : !quiccir.view<16x2x3xf32, "layoutUmod">
    return
  }
}