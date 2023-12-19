// !!! WARNING !!!
// WIP: not lowering, might be an issue with the global const
// !!! WARNING !!!

// ./bin/quiccir-opt ../examples/batched-bwd-standalone.mlir --convert-quiccir-to-affine --lower-affine --arith-bufferize --tensor-bufferize --scf-bufferize --bufferization-bufferize -cse -convert-scf-to-cf -convert-arith-to-llvm  --normalize-memrefs -convert-memref-to-llvm --convert-func-to-llvm -reconcile-unrealized-casts | mlir-cpu-runner -e main -entry-point-result=void --shared-libs=$MLIR_ROOT/lib/libmlir_c_runner_utils.so --shared-libs=$MLIR_ROOT/lib/libmlir_runner_utils.so

memref.global "private" constant @__cst_b_3x2xf32 : memref<3x2xf32> =
dense<[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]>

memref.global "private" constant @__cst_umod_2x2x2xf32 : memref<2x2x2xf32> =
dense<[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]>


!memref_b_t = memref<3x2xf32>

func.func @main() {
  %b0 = memref.get_global @__cst_b_3x2xf32 : memref<3x2xf32>
  %b1 = memref.get_global @__cst_b_3x2xf32 : memref<3x2xf32>
  %umod = memref.get_global @__cst_umod_2x2x2xf32 : memref<2x2x2xf32>

  %b0Ten = bufferization.to_tensor %b0 : memref<3x2xf32>
  %b1Ten = bufferization.to_tensor %b1 : memref<3x2xf32>
  %umodTen = bufferization.to_tensor %umod : memref<2x2x2xf32>

  %d0 = arith.constant 0 : index
  %d1 = arith.constant 1 : index
  %d2 = arith.constant 2 : index
  %E = memref.dim %umod, %d0 : memref<2x2x2xf32>
  %M = memref.dim %umod, %d1 : memref<2x2x2xf32>
  %K = memref.dim %umod, %d2 : memref<2x2x2xf32>

  %zero = arith.constant 0.0 : f32
  %uvalTen = tensor.splat %zero : tensor<2x3x3xf32>

  %t_start = call @rtclock() : () -> f64
  %uvalTenUpdated = affine.for %e = 0 to %E step 1 iter_args(%uv = %uvalTen) -> (tensor<2x3x3xf32>) {
    %umodE = tensor.extract_slice %umodTen[%e, 0, 0][1, 2, 2][1, 1, 1]: tensor<2x2x2xf32> to tensor<2x2xf32>
    %uvalE = quiccir.quadrature %b0Ten, %b1Ten, %umodE : tensor<3x2xf32>, tensor<3x2xf32>, tensor<2x2xf32> -> tensor<3x3xf32>
    %uvUpdated = tensor.insert_slice %uvalE into %uv[%e, 0, 0][1, 3, 3][1, 1, 1]: tensor<3x3xf32> into tensor<2x3x3xf32>
    affine.yield %uvUpdated : tensor<2x3x3xf32>
  }
  %t_end = call @rtclock() : () -> f64

  %t = arith.subf %t_end, %t_start : f64
  call @printF64(%t) : (f64) -> ()
  call @printNewline() : () -> ()

  // %uval = bufferization.to_memref %uvalE : memref<3x3xf32>
  // %unranked = memref.cast %uval : memref<3x3xf32> to memref<*xf32>
  // call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()

  // CHECK: [1,   1,   1],
  // CHECK-NEXT: [1,   1,   1],
  // CHECK-NEXT: [1,   1,   1]

  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
func.func private @printF64(f64)
func.func private @printNewline()
func.func private @rtclock() -> f64



