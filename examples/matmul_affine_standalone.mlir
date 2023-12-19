// llvm 15
// mlir-opt ../examples/matmul_affine_standalone.mlir -inline -lower-affine -convert-scf-to-cf -convert-arith-to-llvm  -convert-memref-to-llvm --convert-func-to-llvm -reconcile-unrealized-casts | mlir-cpu-runner -e main -entry-point-result=void --shared-libs=$MLIR_ROOT/lib/libmlir_c_runner_utils.so --shared-libs=$MLIR_ROOT/lib/libmlir_runner_utils.so


!type_a = memref<16x8xf32>
!type_b = memref<8x8xf32>
!type_c = memref<16x8xf32>
!type_cast = memref<?x?xf32>

func.func private @matmul(%a: !type_cast,
  %b: !type_cast,
  %c: !type_cast) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %a, %c0 : !type_cast
  %K = memref.dim %a, %c1 : !type_cast
  %N = memref.dim %b, %c1 : !type_cast
  affine.for %m = 0 to %M {
    affine.for %n = 0 to %N {
      affine.for %k = 0 to %K {
        %a_v = affine.load %a[%m, %k] : !type_cast
        %b_v = affine.load %b[%k, %n] : !type_cast
        %c_v = affine.load %c[%m, %n] : !type_cast
        %prod = arith.mulf %a_v, %b_v : f32
        %res = arith.addf %prod, %c_v : f32
        affine.store %res, %c[%m, %n] : !type_cast
      }
    }
  }
  return
}

func.func @main() {
  %a = memref.alloc() {alignment = 128} : !type_a
  %b = memref.alloc() {alignment = 128} : !type_b
  %c = memref.alloc() {alignment = 128} : !type_c

  //  Matrix a, b, c dimensions
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %a, %c0 : !type_a
  %K = memref.dim %a, %c1 : !type_a
  %N = memref.dim %b, %c1 : !type_b

  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32

  // Init a
  scf.for %m = %c0 to %M step %c1 {
    scf.for %k = %c0 to %K step %c1 {
      memref.store %one, %a[%m, %k] : !type_a
    }
    memref.store %one, %a[%m, %c0] : !type_a
  }

  // Init b
  scf.for %k = %c0 to %K step %c1 {
    scf.for %n = %c0 to %N step %c1 {
      memref.store %zero, %b[%n, %k] : !type_b
    }
    memref.store %one, %b[%c0, %k] : !type_b
  }

  // Init c
  scf.for %m = %c0 to %M step %c1 {
    scf.for %n = %c0 to %N step %c1 {
      memref.store %zero, %c[%m, %n] : !type_c
    }
  }

  %ac = memref.cast %a : !type_a to !type_cast
  %bc = memref.cast %b : !type_b to !type_cast
  %cc = memref.cast %c : !type_c to !type_cast

  %t_start = call @rtclock() : () -> f64
  call @matmul(%ac, %bc, %cc) : (!type_cast, !type_cast, !type_cast) -> ()
  %t_end = call @rtclock() : () -> f64

  %t = arith.subf %t_end, %t_start : f64
  call @printF64(%t) : (f64) -> ()
  call @printNewline() : () -> ()

  %unranked = memref.cast %c : !type_c to memref<*xf32>
  call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()

  memref.dealloc %a: !type_a
  memref.dealloc %b: !type_b
  memref.dealloc %c: !type_c

  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
func.func private @printF64(f64)
func.func private @printNewline()
func.func private @rtclock() -> f64