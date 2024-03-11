!type_view = !llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>
!type_view_ptr = !llvm.ptr<!type_view>

module {
  func.func private @_ciface_quiccir_jw_prj_layoutUval_layoutUmod(!llvm.ptr, !quiccir.view<1x3x3xf64, "layoutUval">, !quiccir.view<1x2x3xf64, "layoutUmod">)
  func.func private @_ciface_quiccir_jw_int_layoutUmod_layoutUval(!llvm.ptr, !quiccir.view<1x2x3xf64, "layoutUmod">, !quiccir.view<1x3x3xf64, "layoutUval">)
  func.func private @_ciface_quiccir_alloc_jw_prj_layoutUval_layoutUmod(!quiccir.view<1x3x3xf64, "layoutUval">, !quiccir.view<1x2x3xf64, "layoutUmod">)

  func.func @entry(%arg0: !llvm.ptr<array<2 x ptr>>, %arg1: !quiccir.view<1x2x3xf64, "layoutUmod">, %arg2: !quiccir.view<1x2x3xf64, "layoutUmod">) {
    // %0 = quiccir.alloc() : !quiccir.view<1x3x3xf64, "layoutUval">
    %1 = llvm.mlir.undef : !type_view
    %dim0 = llvm.mlir.constant(1 : index) :i32
    %2 = llvm.insertvalue %dim0, %1[0, 2] : !type_view
    %dim1 = llvm.mlir.constant(3 : index) :i32
    %3 = llvm.insertvalue %dim1, %2[0, 1] : !type_view
    %dim2 = llvm.mlir.constant(3 : index) :i32
    %4 = llvm.insertvalue %dim2, %3[0, 0] : !type_view
    %cst = llvm.mlir.constant(1 : index) :i64
    %p0 = llvm.alloca %cst x !type_view {alignment = 8 : i64}: (i64) -> !type_view_ptr
    llvm.store %4, %p0 : !type_view_ptr
    %buf = builtin.unrealized_conversion_cast %p0 : !type_view_ptr to !quiccir.view<1x3x3xf64, "layoutUval">
    func.call @_ciface_quiccir_alloc_jw_prj_layoutUval_layoutUmod(%buf, %arg2) : (!quiccir.view<1x3x3xf64, "layoutUval">, !quiccir.view<1x2x3xf64, "layoutUmod">) -> ()
    %thisArr = llvm.load %arg0: !llvm.ptr<array<2 x ptr>>
    %this1 = llvm.extractvalue %thisArr[0] : !llvm.array<2 x ptr>
    call @_ciface_quiccir_jw_prj_layoutUval_layoutUmod(%this1, %buf, %arg2) : (!llvm.ptr, !quiccir.view<1x3x3xf64, "layoutUval">, !quiccir.view<1x2x3xf64, "layoutUmod">) -> ()
    %this2 = llvm.extractvalue %thisArr[1] : !llvm.array<2 x ptr>
    call @_ciface_quiccir_jw_int_layoutUmod_layoutUval(%this2, %arg1, %buf) : (!llvm.ptr, !quiccir.view<1x2x3xf64, "layoutUmod">, !quiccir.view<1x3x3xf64, "layoutUval">) -> ()
    return
  }
}

// ./bin/quiccir-opt ../examples/alloc-lower-to-call-struct.mlir --finalize-quiccir-view --convert-func-to-llvm --canonicalize | ./bin/quiccir-miniapp -emit=jit --shared-libs=./external/libquiccir_external.so

// ./bin/quiccir-miniapp -emit=jit ../examples/alloc-lower-to-call-struct.mlir