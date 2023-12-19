// RUN: quiccir-opt %s --convert-quiccir-to-affine --bufferization-bufferize -cse -lower-affine -convert-scf-to-cf -convert-arith-to-llvm  -convert-memref-to-llvm --convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext  \
// RUN: -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext    \
// RUN: | FileCheck %s
// RUN: quiccir-opt %s --convert-quiccir-to-linalg --empty-tensor-to-alloc-tensor --linalg-bufferize --bufferization-bufferize -cse --convert-linalg-to-affine-loops -lower-affine -convert-scf-to-cf -convert-arith-to-llvm  -convert-memref-to-llvm --convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext  \
// RUN: -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext    \
// RUN: | FileCheck %s

memref.global "private" constant @__cst_b0_3x2xf32 : memref<3x2xf32> =
dense<[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]>

memref.global "private" constant @__cst_b1_3x2xf32 : memref<3x2xf32> =
dense<[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]>

memref.global "private" constant @__cst_umod_2x2x2xf32 : memref<2x2x2xf32> =
dense<[[[1.0, 0.0], [0.0, 1.0]], [[0.1, 0.0], [0.0, 0.1]]]>

func.func @main() {
  %b0 = memref.get_global @__cst_b0_3x2xf32 : memref<3x2xf32>
  %b1 = memref.get_global @__cst_b1_3x2xf32 : memref<3x2xf32>
  %umod = memref.get_global @__cst_umod_2x2x2xf32 : memref<2x2x2xf32>

  %b0Ten = bufferization.to_tensor %b0 : memref<3x2xf32>
  %b1Ten = bufferization.to_tensor %b1 : memref<3x2xf32>
  %umodTen = bufferization.to_tensor %umod : memref<2x2x2xf32>

  %t_start = call @rtclock() : () -> f64
  %uvalTen = quiccir.quadrature %b0Ten, %b1Ten, %umodTen : tensor<3x2xf32>, tensor<3x2xf32>, tensor<2x2x2xf32> -> tensor<2x3x3xf32>
  %t_end = call @rtclock() : () -> f64

  %t = arith.subf %t_end, %t_start : f64
  call @printF64(%t) : (f64) -> ()
  call @printNewline() : () -> ()

  %uval = bufferization.to_memref %uvalTen : memref<2x3x3xf32>
  %unranked = memref.cast %uval : memref<2x3x3xf32> to memref<*xf32>
  call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()

  // CHECK: [1,   2,   3],
  // CHECK-NEXT: [2,   4,   6],
  // CHECK-NEXT: [3,   6,   9]
  // CHECK-NEXT: [0.1,   0.2,   0.3],
  // CHECK-NEXT: [0.2,   0.4,   0.6],
  // CHECK-NEXT: [0.3,   0.6,   0.9]

  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
func.func private @printF64(f64)
func.func private @printNewline()
func.func private @rtclock() -> f64


// through affine pipeline
// ./bin/quiccir-opt ../test/Quiccir/bwd-standalone.mlir --convert-quiccir-to-affine --bufferization-bufferize -cse -lower-affine -convert-scf-to-cf -convert-arith-to-llvm  -convert-memref-to-llvm --convert-func-to-llvm -reconcile-unrealized-casts | mlir-cpu-runner -e main -entry-point-result=void --shared-libs=$MLIR_ROOT/lib/libmlir_c_runner_utils.so --shared-libs=$MLIR_ROOT/lib/libmlir_runner_utils.so

// through linalg pipeline
// ./bin/quiccir-opt ../test/Quiccir/bwd-standalone.mlir --convert-quiccir-to-linalg --empty-tensor-to-alloc-tensor --linalg-bufferize --bufferization-bufferize -cse --convert-linalg-to-affine-loops -lower-affine -convert-scf-to-cf -convert-arith-to-llvm  -convert-memref-to-llvm --convert-func-to-llvm -reconcile-unrealized-casts | mlir-cpu-runner -e main -entry-point-result=void --shared-libs=$MLIR_ROOT/lib/libmlir_c_runner_utils.so --shared-libs=$MLIR_ROOT/lib/libmlir_runner_utils.so