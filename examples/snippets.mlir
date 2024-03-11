!type_op = memref<1x3x2xf64>
!type_umod = memref<1x2x3xf64>
!type_uval = memref<1x3x3xf64>

!type_top = tensor<1x3x2xf64>
!type_tumod = tensor<1x2x3xf64>
!type_tuval = tensor<1x3x3xf64>
!type_tcast = tensor<?x?x?xf64>

// mod -> phys
func.func @wrap_batched_op(%this0: !llvm.ptr, %this1: !llvm.ptr, %uval: !type_uval, %op: !type_op, %umod: !type_umod)
  attributes {llvm.emit_c_interface} {
  %top = bufferization.to_tensor %op : !type_op
  %tumod = bufferization.to_tensor %umod : !type_umod



  %ret = "qir.al.prj.p"(%tumod) {ifobj = 0 : i64, dim = 3 : i64}: (!type_tumod) -> !type_tuval
// can lower directly to a call


//   %val = quiccir.quadrature %top, %tumod : !type_top, !type_tumod -> !type_tuval

//   call @qir_al_p(%this0, %cast, %cast_0, %cast_1)


  %tmpref = bufferization.to_memref %ret : !type_uval
  memref.copy %tmpref, %uval : !type_uval to !type_uval
  return
}

// ./bin/quiccir-opt ../examples/wrap_batched_call.mlir --convert-quiccir-to-call --convert-memref-to-llvm --convert-func-to-llvm -cse --reconcile-unrealized-casts | ./bin/quiccir-miniapp -emit=jit --shared-libs=./external/libquiccir_external.so

// ./bin/quiccir-opt ../examples/wrap_batched_call.mlir --convert-quiccir-to-call --canonicalize --convert-memref-to-llvm --convert-func-to-llvm --canonicalize | ./bin/quiccir-miniapp -emit=jit --shared-libs=./external/libquiccir_external.so