func.func private @matmul(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>) {
    affine.for %arg3 = 0 to 128 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 128 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<256x256xf32>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<256x256xf32>
          %2 = affine.load %arg2[%arg3, %arg4] : memref<256x256xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %3, %2 : f32
          affine.store %4, %arg2[%arg3, %arg4] : memref<256x256xf32>
        }
      }
    }
    return
}

// naive
// mlir-opt ../examples/matmul-parallel-1d.mlir -affine-parallelize -lower-affine -canonicalize --scf-parallel-loop-tiling="parallel-loop-tile-sizes=16,16 no-min-max-bounds" -cse


// blocked
// mlir-opt ../examples/matmul-parallel-1d.mlir --affine-loop-tile="tile-size=16" --test-loop-permutation="permutation-map=0,1,4,3,2,5" --affine-parallelize -lower-affine -cse -canonicalize --loop-invariant-code-motion -canonicalize
// --gpu-kernel-outlining{data-layout-str=<string>}

// --pass-pipeline="builtin.module(func.func(affine-scalrep, affine-parallelize, lower-affine, canonicalize))"