
!type_view = !llvm.ptr<!llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>>

// mod -> phys
module {
  func.func private @_ciface_quiccir_jw_prj_layoutUval_layoutUmod(!llvm.ptr, !type_view, !type_view)
  func.func @entry(%thisArr: !llvm.array<1x!llvm.ptr>, %uval: !type_view, %umod: !type_view) {
    %this = llvm.extractvalue %thisArr [0] : !llvm.array<1x!llvm.ptr>
    call @_ciface_quiccir_jw_prj_layoutUval_layoutUmod(%this, %uval, %umod) : (!llvm.ptr, !type_view, !type_view) -> ()
    return
  }
}


// ./bin/quiccir-opt ../examples/jw-call.mlir --convert-func-to-llvm --canonicalize | ./bin/quiccir-miniapp -emit=jit --shared-libs=./external/libquiccir_external.so
