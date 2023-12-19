// llvm 16

// this has a stupid mapping but works
// mlir-opt ../examples/matmul_affine_standalone_gpu.mlir --affine-parallelize -lower-affine -cse -gpu-map-parallel-loops -convert-parallel-loops-to-gpu --canonicalize  --convert-scf-to-cf --convert-vector-to-llvm --convert-arith-to-llvm --convert-memref-to-llvm -lower-affine -gpu-kernel-outlining | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,reconcile-unrealized-casts,gpu-to-cubin))' | mlir-opt --gpu-to-llvm --reconcile-unrealized-casts | mlir-cpu-runner --entry-point-result=void --shared-libs=$MLIR_ROOT/lib/libmlir_cuda_runtime.so --shared-libs=$MLIR_ROOT/lib/libmlir_c_runner_utils.so --shared-libs=$MLIR_ROOT/lib/libmlir_runner_utils.so


// 2D block only
// mlir-opt ../examples/matmul_affine_standalone_gpu.mlir --affine-parallelize -lower-affine -canonicalize -cse -gpu-map-parallel-loops -convert-parallel-loops-to-gpu --canonicalize  --convert-scf-to-cf --convert-arith-to-llvm --convert-memref-to-llvm -gpu-kernel-outlining | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,reconcile-unrealized-casts,gpu-to-cubin{chip=sm_80}))' | mlir-opt --gpu-to-llvm --reconcile-unrealized-casts | ncu --set full -f --page details --details-all -o naive_mat mlir-cpu-runner --entry-point-result=void --shared-libs=$MLIR_ROOT/lib/libmlir_cuda_runtime.so --shared-libs=$MLIR_ROOT/lib/libmlir_c_runner_utils.so --shared-libs=$MLIR_ROOT/lib/libmlir_runner_utils.so

// 2D block and threads (classic naive mapping)
// reset && mlir-opt ../examples/matmul_affine_standalone_gpu.mlir --affine-parallelize -lower-affine -canonicalize --scf-parallel-loop-tiling="parallel-loop-tile-sizes=16,16 no-min-max-bounds" -canonicalize -gpu-map-parallel-loops -convert-parallel-loops-to-gpu --convert-scf-to-cf -gpu-kernel-outlining | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,lower-affine,convert-gpu-to-nvvm,convert-index-to-llvm,reconcile-unrealized-casts,gpu-to-cubin{chip=sm_80}))'| mlir-opt --inline --convert-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-index-to-llvm --gpu-to-llvm --reconcile-unrealized-casts | ncu --set full -f --page details --details-all -o naive_mat mlir-cpu-runner --entry-point-result=void --shared-libs=$MLIR_ROOT/lib/libmlir_cuda_runtime.so --shared-libs=$MLIR_ROOT/lib/libmlir_c_runner_utils.so --shared-libs=$MLIR_ROOT/lib/libmlir_runner_utils.so


// WIP manual tile/permutation
// mlir-opt ../examples/matmul_affine_standalone_gpu.mlir --affine-loop-tile="tile-size=16" --test-loop-permutation="permutation-map=0,1,4,3,2,5" --affine-parallelize -lower-affine -cse -canonicalize --loop-invariant-code-motion -canonicalize -convert-parallel-loops-to-gpu --convert-scf-to-cf --convert-arith-to-llvm --convert-memref-to-llvm -gpu-kernel-outlining | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,reconcile-unrealized-casts,gpu-to-cubin{chip=sm_80}))' | mlir-opt --gpu-to-llvm --reconcile-unrealized-casts | mlir-cpu-runner --entry-point-result=void --shared-libs=$MLIR_ROOT/lib/libmlir_cuda_runtime.so --shared-libs=$MLIR_ROOT/lib/libmlir_c_runner_utils.so --shared-libs=$MLIR_ROOT/lib/libmlir_runner_utils.so

!type_a = memref<256x256xf32>
!type_b = memref<256x256xf32>
!type_c = memref<256x256xf32>
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

  %gpu_ac = memref.cast %gpu_a : !type_a to !type_cast
  %gpu_bc = memref.cast %gpu_b : !type_b to !type_cast
  %gpu_cc = memref.cast %gpu_c : !type_c to !type_cast

  %t_start = call @rtclock() : () -> f64
  call @matmul(%gpu_ac, %gpu_bc, %gpu_cc) : (!type_cast, !type_cast, !type_cast) -> ()
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