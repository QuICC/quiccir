!type_view = !llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>
!type_view_ptr = !llvm.ptr<!type_view>

module {
  func.func private @_ciface_quiccir_jw_prj_layoutUval_layoutUmod(!llvm.ptr, !quiccir.view<1x3x3xf64, "layoutUval">, !quiccir.view<1x2x3xf64, "layoutUmod">)
  func.func private @_ciface_quiccir_jw_int_layoutUmod_layoutUval(!llvm.ptr, !quiccir.view<1x2x3xf64, "layoutUmod">, !quiccir.view<1x3x3xf64, "layoutUval">)

  func.func @entry(%arg0: !llvm.ptr<array<2 x ptr>>, %arg1: !quiccir.view<1x2x3xf64, "layoutUmod">, %arg2: !quiccir.view<1x2x3xf64, "layoutUmod">) {
    %buf = quiccir.alloc(%arg2) : !quiccir.view<1x2x3xf64, "layoutUmod"> -> !quiccir.view<1x3x3xf64, "layoutUval"> {producer="quiccir.jw.prj"}
    %thisArr = llvm.load %arg0: !llvm.ptr<array<2 x ptr>>
    %this1 = llvm.extractvalue %thisArr[0] : !llvm.array<2 x ptr>
    call @_ciface_quiccir_jw_prj_layoutUval_layoutUmod(%this1, %buf, %arg2) : (!llvm.ptr, !quiccir.view<1x3x3xf64, "layoutUval">, !quiccir.view<1x2x3xf64, "layoutUmod">) -> ()
    %this2 = llvm.extractvalue %thisArr[1] : !llvm.array<2 x ptr>
    call @_ciface_quiccir_jw_int_layoutUmod_layoutUval(%this2, %arg1, %buf) : (!llvm.ptr, !quiccir.view<1x2x3xf64, "layoutUmod">, !quiccir.view<1x3x3xf64, "layoutUval">) -> ()
    return
  }
}

// ./bin/quiccir-opt ../examples/alloc-lower-to-call-view.mlir --lower-quiccir-alloc --finalize-quiccir-view --convert-func-to-llvm --canonicalize | ./bin/quiccir-miniapp -emit=jit --shared-libs=./external/libquiccir_external.so

// ./bin/quiccir-miniapp -emit=jit ../examples/alloc-lower-to-call-view.mlir