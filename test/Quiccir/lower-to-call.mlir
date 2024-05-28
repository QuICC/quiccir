// RUN: quiccir-opt %s --convert-quiccir-to-call | FileCheck %s

module {
    // materialize to existing buffer
    func.func @entryJwPrjBuf(%thisArr: !llvm.ptr<array<1 x ptr>>, %vumod: !quiccir.view<16x2x3xf32, "layoutUmod">) {
    %vuval = quiccir.alloc(%vumod): !quiccir.view<16x2x3xf32, "layoutUmod"> -> !quiccir.view<16x3x3xf32, "layoutUval"> {producer = "quiccir.jw.prj"}
    %umod = builtin.unrealized_conversion_cast %vumod : !quiccir.view<16x2x3xf32, "layoutUmod"> to tensor<16x2x3xf32, "layoutUmod">
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<array<1 x ptr>>
    // CHECK: %{{.*}} = llvm.extractvalue %{{.*}}[0] : !llvm.array<1 x ptr>
    // CHECK: call @_ciface_quiccir_jw_prj_f32_layoutUval_f32_layoutUmod(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<16x3x3xf32, "layoutUval">, !quiccir.view<16x2x3xf32, "layoutUmod">) -> ()
    %ret = quiccir.jw.prj %umod : tensor<16x2x3xf32, "layoutUmod"> -> tensor<16x3x3xf32, "layoutUval"> attributes{implptr = 0 :i64}
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !quiccir.view<16x3x3xf32, "layoutUval"> to tensor<16x3x3xf32, "layoutUval">
    quiccir.materialize %ret in %vuval : (tensor<16x3x3xf32, "layoutUval">, !quiccir.view<16x3x3xf32, "layoutUval">)
    return
    }

    // create new buffer
    func.func @entryJwPrjAlloc(%thisArr: !llvm.ptr<array<2 x ptr>>, %vumod: !quiccir.view<16x2x3xf32, "layoutUmod">) {
    %umod = builtin.unrealized_conversion_cast %vumod : !quiccir.view<16x2x3xf32, "layoutUmod"> to tensor<16x2x3xf32, "layoutUmod">
    // CHECK: %{{.*}} = quiccir.alloc(%{{.*}}) : !quiccir.view<16x2x3xf32, "layoutUmod"> -> !quiccir.view<16x3x3xf32, "layoutUval"> {producer = "quiccir.jw.prj"}
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<array<2 x ptr>>
    // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x ptr>
    // CHECK: call @_ciface_quiccir_jw_prj_f32_layoutUval_f32_layoutUmod(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<16x3x3xf32, "layoutUval">, !quiccir.view<16x2x3xf32, "layoutUmod">) -> ()
    %ret = quiccir.jw.prj %umod : tensor<16x2x3xf32, "layoutUmod"> -> tensor<16x3x3xf32, "layoutUval"> attributes{implptr = 1 :i64}
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !quiccir.view<16x3x3xf32, "layoutUval"> to tensor<16x3x3xf32, "layoutUval">
    return
    }

    // create new buffer
    func.func @entryFrPrjAlloc(%thisArr: !llvm.ptr<array<2 x ptr>>, %vumod: !quiccir.view<16x2x3xf32, "layoutUmod">) {
    %umod = builtin.unrealized_conversion_cast %vumod : !quiccir.view<16x2x3xf32, "layoutUmod"> to tensor<16x2x3xf32, "layoutUmod">
    // CHECK: %{{.*}} = quiccir.alloc(%{{.*}}) : !quiccir.view<16x2x3xf32, "layoutUmod"> -> !quiccir.view<16x3x3xf32, "layoutUval"> {producer = "quiccir.fr.prj"}
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<array<2 x ptr>>
    // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x ptr>
    // CHECK: call @_ciface_quiccir_fr_prj_f32_layoutUval_f32_layoutUmod(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<16x3x3xf32, "layoutUval">, !quiccir.view<16x2x3xf32, "layoutUmod">) -> ()
    %ret = quiccir.fr.prj %umod : tensor<16x2x3xf32, "layoutUmod"> -> tensor<16x3x3xf32, "layoutUval"> attributes{implptr = 1 :i64}
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !quiccir.view<16x3x3xf32, "layoutUval"> to tensor<16x3x3xf32, "layoutUval">
    return
    }

    // materialize to existing buffer
    func.func @entryTransposeBuf(%thisArr: !llvm.ptr<array<1 x ptr>>, %v: !quiccir.view<16x2x3xf32, "layoutIn">, %vtra: !quiccir.view<16x3x2xf32, "layoutOut">) {
    %vt = builtin.unrealized_conversion_cast %v : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<array<1 x ptr>>
    // CHECK: %{{.*}} = llvm.extractvalue %{{.*}}[0] : !llvm.array<1 x ptr>
    // CHECK: call @_ciface_quiccir_transpose_021_f32_layoutOut_f32_layoutIn(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<16x3x2xf32, "layoutOut">, !quiccir.view<16x2x3xf32, "layoutIn">) -> ()
    %tra = quiccir.transpose %vt permutation = [0, 2, 1] : tensor<16x2x3xf32, "layoutIn"> -> tensor<16x3x2xf32, "layoutOut"> attributes{implptr = 0 :i64}
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !quiccir.view<16x3x2xf32, "layoutOut"> to tensor<16x3x2xf32, "layoutOut">
    quiccir.materialize %tra in %vtra : (tensor<16x3x2xf32, "layoutOut">, !quiccir.view<16x3x2xf32, "layoutOut">)
    return
    }

    // create new buffer
    func.func @entryTransposeAlloc(%thisArr: !llvm.ptr<array<1 x ptr>>, %v: !quiccir.view<16x2x3xf32, "layoutIn">) {
    %vt = builtin.unrealized_conversion_cast %v : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    // CHECK: %{{.*}} = quiccir.alloc(%{{.*}}) : !quiccir.view<16x2x3xf32, "layoutIn"> -> !quiccir.view<16x3x2xf32, "layoutOut"> {producer = "quiccir.transpose_021"}
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<array<1 x ptr>>
    // CHECK: %{{.*}} = llvm.extractvalue %{{.*}}[0] : !llvm.array<1 x ptr>
    // CHECK: call @_ciface_quiccir_transpose_021_f32_layoutOut_f32_layoutIn(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<16x3x2xf32, "layoutOut">, !quiccir.view<16x2x3xf32, "layoutIn">) -> ()
    %tra = quiccir.transpose %vt permutation = [0, 2, 1] : tensor<16x2x3xf32, "layoutIn"> -> tensor<16x3x2xf32, "layoutOut"> attributes{implptr = 0 :i64}
    return
    }

    // create new buffer
    func.func @entrySubAlloc(%thisArr: !llvm.ptr<array<1 x ptr>>, %a: !quiccir.view<16x2x3xf32, "layoutIn">, %b: !quiccir.view<16x2x3xf32, "layoutIn">) {
    %at = builtin.unrealized_conversion_cast %a : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    %bt = builtin.unrealized_conversion_cast %b : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    // CHECK: %{{.*}} = quiccir.alloc(%{{.*}}) : !quiccir.view<16x2x3xf32, "layoutIn"> -> !quiccir.view<16x2x3xf32, "layoutIn"> {producer = "quiccir.sub"}
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<array<1 x ptr>>
    // CHECK: %[[THIS:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<1 x ptr>
    // CHECK: call @_ciface_quiccir_sub_f32_layoutIn_f32_layoutIn_f32_layoutIn(%[[THIS]], %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">) -> ()
    %tra = quiccir.sub %at, %bt : tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn"> -> tensor<16x2x3xf32, "layoutIn"> attributes{implptr = 0 :i64}
    return
    }

    // create new buffer
    func.func @entryAddAlloc(%thisArr: !llvm.ptr<array<1 x ptr>>, %a: !quiccir.view<16x2x3xf32, "layoutIn">, %b: !quiccir.view<16x2x3xf32, "layoutIn">) {
    %at = builtin.unrealized_conversion_cast %a : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    %bt = builtin.unrealized_conversion_cast %b : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    // CHECK: %{{.*}} = quiccir.alloc(%{{.*}}) : !quiccir.view<16x2x3xf32, "layoutIn"> -> !quiccir.view<16x2x3xf32, "layoutIn"> {producer = "quiccir.add"}
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<array<1 x ptr>>
    // CHECK: %[[THIS:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<1 x ptr>
    // CHECK: call @_ciface_quiccir_add_f32_layoutIn_f32_layoutIn_f32_layoutIn(%[[THIS]], %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !quiccir.view<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">, !quiccir.view<16x2x3xf32, "layoutIn">) -> ()
    %tra = quiccir.add %at, %bt : tensor<16x2x3xf32, "layoutIn">, tensor<16x2x3xf32, "layoutIn"> -> tensor<16x2x3xf32, "layoutIn"> attributes{implptr = 0 :i64}
    return
    }
}
