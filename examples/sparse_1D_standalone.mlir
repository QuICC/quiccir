// RUN: quiccir-opt %s --convert-quiccir-to-affine --bufferization-bufferize -cse -lower-affine -convert-scf-to-cf -convert-arith-to-llvm  -convert-memref-to-llvm --convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext  \
// RUN: -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext    \
// RUN: | FileCheck %s

memref.global "private" constant @__cst_pos_2xi32 : memref<2xi32> =
dense<[0, 3]>
memref.global "private" constant @__cst_coo_3xi32 : memref<3xi32> =
dense<[0, 2, 4]>
memref.global "private" constant @__cst_val_3xf32 : memref<3xf32> =
dense<[0.0, 1.0, 2.0]>

#SP = #sparse_tensor.encoding<{ lvlTypes = ["compressed"]}>

func.func @main() {
  %pos = memref.get_global @__cst_pos_2xi32 : memref<2xi32>
  %coo = memref.get_global @__cst_coo_3xi32 : memref<3xi32>
  %val = memref.get_global @__cst_val_3xf32 : memref<3xf32>

  %posTen = bufferization.to_tensor %pos : memref<2xi32>
  %cooTen = bufferization.to_tensor %coo : memref<3xi32>
  %valTen = bufferization.to_tensor %val : memref<3xf32>

  %st = sparse_tensor.assemble %valTen, %posTen, %cooTen
    : tensor<3xf32>, tensor<2xi32>, tensor<3xi32> to tensor<8xf32, #SP>

  // %d0 = arith.constant 0 : index
  // %d1 = arith.constant 1 : index
  // %d2 = arith.constant 2 : index
  // %E = memref.dim %umod, %d0 : memref<2x2x2xf32>
  // %M = memref.dim %umod, %d1 : memref<2x2x2xf32>
  // %K = memref.dim %umod, %d2 : memref<2x2x2xf32>

  // %zero = arith.constant 0.0 : f32
  // %uvalTen = tensor.splat %zero : tensor<2x3x3xf32>

  // %t_start = call @rtclock() : () -> f64
  // affine.for %e = 0 to %E {
  //   %umod_e = memref.subview %umod[%e, 0, 0][1, 2, 2][1, 1, 1] : memref<2x2x2xf32> to memref<2x2xf32, offset: ?, strides: [2, 1]>
  //   %umodTen_e = bufferization.to_tensor %umod_e : memref<2x2xf32, offset: ?, strides: [2, 1]>
  //   %uvalTen_e = quiccir.quadrature %b0Ten, %b1Ten, %umodTen_e : tensor<3x2xf32>, tensor<3x2xf32>, tensor<2x2xf32> -> tensor<3x3xf32>
  //   %uval_e = bufferization.to_memref %uvalTen_e : memref<3x3xf32>
  //   %uvalTmp_e = memref.subview %uval[%e, 0, 0][1, 3, 3][1, 1, 1] : memref<2x3x3xf32> to memref<3x3xf32, offset: ?, strides: [3, 1]>
  //   memref.copy %uval_e, %uvalTmp_e : memref<3x3xf32> to memref<3x3xf32, offset: ?, strides: [3, 1]>
  // }
  // %t_end = call @rtclock() : () -> f64

  // %t = arith.subf %t_end, %t_start : f64
  // call @printF64(%t) : (f64) -> ()
  // call @printNewline() : () -> ()

  // %unranked = memref.cast %uval : memref<2x3x3xf32> to memref<*xf32>
  // call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()

  // // CHECK: [1,   1,   1],
  // // CHECK-NEXT: [1,   1,   1],
  // // CHECK-NEXT: [1,   1,   1]

  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
func.func private @printF64(f64)
func.func private @printNewline()
func.func private @rtclock() -> f64


// quiccir-opt ../examples/batched-memref-bwd-standalone.mlir --convert-quiccir-to-affine --bufferization-bufferize -cse -lower-affine -convert-scf-to-cf -convert-arith-to-llvm  -convert-memref-to-llvm --convert-func-to-llvm -reconcile-unrealized-casts | mlir-cpu-runner -e main -entry-point-result=void --shared-libs=$MLIR_ROOT/lib/libmlir_c_runner_utils.so --shared-libs=$MLIR_ROOT/lib/libmlir_runner_utils.so
