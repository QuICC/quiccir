// this is what we can lowe to but it needs to "merge" the parallel loops

module {
  func private @matmul(%arg0: memref<16x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<16x8xf32>) {
    affine.parallel (%arg3) = (0) to (16) {
      affine.parallel (%arg4) = (0) to (8) {
        affine.for %arg5 = 0 to 8 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<16x8xf32>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<8x8xf32>
          %2 = affine.load %arg2[%arg3, %arg4] : memref<16x8xf32>
          %3 = mulf %0, %1 : f32
          %4 = addf %3, %2 : f32
          affine.store %4, %arg2[%arg3, %arg4] : memref<16x8xf32>
        }
      }
    }
    return
  }
  func @main() {
    %c8 = constant 8 : index
    %c16 = constant 16 : index
    %cst = constant 1.000000e+00 : f32
    %cst_0 = constant 0.000000e+00 : f32
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<16x8xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<8x8xf32>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<16x8xf32>
    scf.for %arg0 = %c0 to %c16 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        memref.store %cst, %0[%arg0, %arg1] : memref<16x8xf32>
      }
      memref.store %cst, %0[%arg0, %c0] : memref<16x8xf32>
    }
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        memref.store %cst_0, %1[%arg1, %arg0] : memref<8x8xf32>
      }
      memref.store %cst, %1[%c0, %arg0] : memref<8x8xf32>
    }
    scf.for %arg0 = %c0 to %c16 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        memref.store %cst_0, %2[%arg0, %arg1] : memref<16x8xf32>
      }
    }
    %3 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%3] () : memref<16x8xf32>
    %memref_1, %asyncToken_2 = gpu.alloc async [%3] () : memref<8x8xf32>
    %memref_3, %asyncToken_4 = gpu.alloc async [%3] () : memref<16x8xf32>
    %4 = gpu.memcpy async [%3] %memref, %0 : memref<16x8xf32>, memref<16x8xf32>
    %5 = gpu.memcpy async [%3] %memref_1, %1 : memref<8x8xf32>, memref<8x8xf32>
    %6 = gpu.memcpy async [%3] %memref_3, %2 : memref<16x8xf32>, memref<16x8xf32>
    %7 = call @rtclock() : () -> f64
    call @matmul(%memref, %memref_1, %memref_3) : (memref<16x8xf32>, memref<8x8xf32>, memref<16x8xf32>) -> ()
    %8 = call @rtclock() : () -> f64
    %9 = subf %8, %7 : f64
    call @printF64(%9) : (f64) -> ()
    call @printNewline() : () -> ()
    %10 = gpu.wait async
    %11 = gpu.memcpy async [%10] %2, %memref_3 : memref<16x8xf32>, memref<16x8xf32>
    gpu.wait [%11]
    %12 = memref.cast %2 : memref<16x8xf32> to memref<*xf32>
    call @printMemrefF32(%12) : (memref<*xf32>) -> ()
    memref.dealloc %0 : memref<16x8xf32>
    memref.dealloc %1 : memref<8x8xf32>
    memref.dealloc %2 : memref<16x8xf32>
    return
  }
  func private @printMemrefF32(memref<*xf32>)
  func private @printF64(f64)
  func private @printNewline()
  func private @rtclock() -> f64
}

