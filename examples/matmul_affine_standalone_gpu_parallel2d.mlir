// llvm 15
// mlir-opt ../examples/matmul_affine_standalone_gpu_parallel2d.mlir -lower-affine -cse -gpu-map-parallel-loops -convert-parallel-loops-to-gpu --canonicalize  --convert-scf-to-cf --convert-vector-to-llvm --convert-arith-to-llvm --convert-memref-to-llvm -lower-affine -gpu-kernel-outlining -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,reconcile-unrealized-casts,gpu-to-cubin)' --gpu-to-llvm --reconcile-unrealized-casts | mlir-cpu-runner --entry-point-result=void --shared-libs=$MLIR_ROOT/lib/libmlir_cuda_runtime.so --shared-libs=$MLIR_ROOT/lib/libmlir_c_runner_utils.so --shared-libs=$MLIR_ROOT/lib/libmlir_runner_utils.so

!type_a = memref<16x8xf32>
!type_b = memref<8x8xf32>
!type_c = memref<16x8xf32>
!type_cast = memref<?x?xf32>

func.func private @matmul(%a: !type_a,
  %b: !type_b,
  %c: !type_c) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %a, %c0 : !type_a
  %K = memref.dim %a, %c1 : !type_a
  %N = memref.dim %b, %c1 : !type_b
  affine.parallel (%m, %n) = (0, 0) to (%M, %N) {
      affine.for %k = 0 to %K {
        %a_v = affine.load %a[%m, %k] : !type_a
        %b_v = affine.load %b[%k, %n] : !type_b
        %c_v = affine.load %c[%m, %n] : !type_c
        %prod = arith.mulf %a_v, %b_v : f32
        %res = arith.addf %prod, %c_v : f32
        affine.store %res, %c[%m, %n] : !type_c
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

  %t0 = gpu.wait async

  // Allocate actual input/output arrays on device
  %gpu_a, %t5 = gpu.alloc async [%t0] () : !type_a
  %gpu_b, %t6 = gpu.alloc async [%t0] () : !type_b
  %gpu_c, %t7 = gpu.alloc async [%t0] () : !type_c

  // Copy initialized arrays from host to device
  %t2 = gpu.memcpy async [%t0] %gpu_a, %a : !type_a, !type_a
  %t3 = gpu.memcpy async [%t0] %gpu_b, %b : !type_b, !type_b
  %t4 = gpu.memcpy async [%t0] %gpu_c, %c : !type_c, !type_c

  gpu.wait [%t0]

  %t_start = call @rtclock() : () -> f64
  call @matmul(%gpu_a, %gpu_b, %gpu_c) : (!type_a, !type_b, !type_c) -> ()
  %t_end = call @rtclock() : () -> f64

  %t = arith.subf %t_end, %t_start : f64
  call @printF64(%t) : (f64) -> ()
  call @printNewline() : () -> ()

  %t1 = gpu.wait async
  // Copy result matrix back to host for printing.
  %t8 = gpu.memcpy async [%t1] %c, %gpu_c : !type_c, !type_c
  gpu.wait[%t8]

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