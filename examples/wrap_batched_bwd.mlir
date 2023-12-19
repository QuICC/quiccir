!type_b0 = memref<3x2xf64>
!type_b1 = memref<3x2xf64>
!type_umod = memref<2x2x2xf64>
!type_uval = memref<3x3x2xf64>

!type_tb0 = tensor<3x2xf64>
!type_tb1 = tensor<3x2xf64>
!type_tumod = tensor<2x2x2xf64>
!type_tuval = tensor<3x3x2xf64>
!type_tcast = tensor<?x?x?xf64>

// func.func private @matmul(%a: !type_cast, %b: !type_cast) -> !type_cast {
//   %dim0 = arith.constant 0 : index
//   %dim1 = arith.constant 1 : index
//   %M = tensor.dim %a, %dim0 : !type_cast
//   %N = tensor.dim %b, %dim1 : !type_cast
//   %0 = tensor.empty(%M, %N) : !type_cast
//   %c = linalg.matmul ins(%a, %b : !type_cast, !type_cast) outs(%0 : !type_cast) -> !type_cast
//   return %c : !type_cast
// }

func.func @wrap_batched_bwd(%b0: !type_b0,
  %b1: !type_b1, %umod: !type_umod, %uval: !type_uval)
  attributes {llvm.emit_c_interface} {
  // %ta = bufferization.to_tensor %a : !type_a
  // %tb = bufferization.to_tensor %b : !type_b
  // %cast_ta = tensor.cast %ta : !type_ta to tensor<?x?xf64>
  // %cast_tb = tensor.cast %tb : !type_tb to tensor<?x?xf64>
  // %ret = call @matmul(%cast_ta, %cast_tb) : (!type_cast, !type_cast) -> (!type_cast)
  // %cast_ret = tensor.cast %ret : !type_cast to !type_tc
  // %tmpref = bufferization.to_memref %cast_ret : !type_c
  // memref.copy %tmpref, %c : !type_c to !type_c
  return
}

// ./bin/quiccir-miniapp -emit=jit ../examples/wrap_batched_bwd.mlir