!type_a = memref<4x1xf64, strided<[1, 4]>>
!type_b = memref<1x2xf64, strided<[1, 1]>>
!type_c = memref<4x2xf64, strided<[1, 4]>>

!type_ta = tensor<4x1xf64>
!type_tb = tensor<1x2xf64>
!type_tc = tensor<4x2xf64>
!type_cast = tensor<?x?xf64>

func.func private @matmul(%a: !type_cast, %b: !type_cast) -> !type_cast {
  %dim0 = arith.constant 0 : index
  %dim1 = arith.constant 1 : index
  %M = tensor.dim %a, %dim0 : !type_cast
  %N = tensor.dim %b, %dim1 : !type_cast
  %0 = tensor.empty(%M, %N) : !type_cast
  %c = linalg.matmul ins(%a, %b : !type_cast, !type_cast) outs(%0 : !type_cast) -> !type_cast
  return %c : !type_cast
}

func.func @wrap_matmul(%a: !type_a,
  %b: !type_b,
  %c: !type_c) {
  %ta = bufferization.to_tensor %a : !type_a
  %tb = bufferization.to_tensor %b : !type_b
  %cast_ta = tensor.cast %ta : !type_ta to tensor<?x?xf64>
  %cast_tb = tensor.cast %tb : !type_tb to tensor<?x?xf64>
  %ret = call @matmul(%cast_ta, %cast_tb) : (!type_cast, !type_cast) -> (!type_cast)
  %cast_ret = tensor.cast %ret : !type_cast to !type_tc
  %tmpref = bufferization.to_memref %cast_ret : !type_c
  memref.copy %tmpref, %c : !type_c to !type_c
  return
}

// mlir-opt ../examples/matmul_tensor.mlir -inline -linalg-comprehensive-module-bufferize -convert-linalg-to-affine-loops -lower-affine -convert-memref-to-llvm -convert-scf-to-std -convert-arith-to-llvm -convert-std-to-llvm="use-bare-ptr-memref-call-conv" -reconcile-unrealized-casts
