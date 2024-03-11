// RUN: quiccir-opt %s --convert-quiccir-to-call | FileCheck %s

module {
    // materialize to existing buffer
    func.func @entry0(%thisArr: !llvm.ptr<array<1 x ptr>>, %vumod: !quiccir.view<16x2x3xf32, "layoutUmod">) {
    %vuval = quiccir.alloc(%vumod): !quiccir.view<16x2x3xf32, "layoutUmod"> -> !quiccir.view<16x3x3xf32, "layoutUval"> {producer = "quiccir.jw.prj"}
    %umod = builtin.unrealized_conversion_cast %vumod : !quiccir.view<16x2x3xf32, "layoutUmod"> to tensor<16x2x3xf32, "layoutUmod">
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<array<1 x ptr>>
    // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<1 x ptr>
    // CHECK: call @_ciface_quiccir_jw_prj_layoutUval_layoutUmod(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<16x3x3xf32, "layoutUval">, !quiccir.view<16x2x3xf32, "layoutUmod">) -> ()
    %ret = quiccir.jw.prj %umod : tensor<16x2x3xf32, "layoutUmod"> -> tensor<16x3x3xf32, "layoutUval"> attributes{implptr = 0 :i64}
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !quiccir.view<16x3x3xf32, "layoutUval"> to tensor<16x3x3xf32, "layoutUval">
    quiccir.materialize %ret in %vuval : (tensor<16x3x3xf32, "layoutUval">, !quiccir.view<16x3x3xf32, "layoutUval">)
    return
    }

    // create new buffer
    func.func @entry1(%thisArr: !llvm.ptr<array<2 x ptr>>, %vumod: !quiccir.view<16x2x3xf32, "layoutUmod">) {
    %umod = builtin.unrealized_conversion_cast %vumod : !quiccir.view<16x2x3xf32, "layoutUmod"> to tensor<16x2x3xf32, "layoutUmod">
    // CHECK: %{{.*}} = quiccir.alloc(%{{.*}}) : !quiccir.view<16x2x3xf32, "layoutUmod"> -> !quiccir.view<16x3x3xf32, "layoutUval"> {producer = "quiccir.jw.prj"}
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<array<2 x ptr>>
    // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x ptr>
    // CHECK: call @_ciface_quiccir_jw_prj_layoutUval_layoutUmod(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<16x3x3xf32, "layoutUval">, !quiccir.view<16x2x3xf32, "layoutUmod">) -> ()
    %ret = quiccir.jw.prj %umod : tensor<16x2x3xf32, "layoutUmod"> -> tensor<16x3x3xf32, "layoutUval"> attributes{implptr = 1 :i64}
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !quiccir.view<16x3x3xf32, "layoutUval"> to tensor<16x3x3xf32, "layoutUval">
    return
    }
}