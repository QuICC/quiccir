// mod -> phys
!type_umod = !quiccir.view<1x2x3xf64, "layoutUmod">
!type_uval = !quiccir.view<1x3x3xf64, "layoutUval">

!type_tumod = tensor<1x2x3xf64, "layoutUmod">
!type_tuval = tensor<1x3x3xf64, "layoutUval">

func.func @entry(%thisArr: !llvm.ptr<array<2 x ptr>> {llvm.noalias}, %uout: !type_umod, %umod: !type_umod) {
  %tumod = builtin.unrealized_conversion_cast %umod : !type_umod to !type_tumod
  /// with multiple ops, allocate new buffer
  %tuval = quiccir.jw.prj %tumod : !type_tumod -> !type_tuval attributes{implptr = 0 :i64}
  %ret = quiccir.jw.int %tuval : !type_tuval -> !type_tumod attributes{implptr = 1 :i64}
  /// if this is the only consumer write to existing buffer
  quiccir.materialize %ret in %uout : (!type_tumod, !type_umod)
  return
}

// ./bin/quiccir-opt ../examples/jw-lower-to-lib-call-2ops.mlir --convert-quiccir-to-call --quiccir-view-deallocation --lower-quiccir-alloc --canonicalize --finalize-quiccir-view --convert-func-to-llvm --canonicalize | ./bin/quiccir-miniapp -emit=jit --shared-libs=./external/libquiccir_external.so

// or
// ./bin/quiccir-miniapp ../examples/jw-lower-to-lib-call-2ops.mlir --emit=jit -opt

//--buffer-deallocation