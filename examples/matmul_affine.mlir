!type_a = memref<4x1xf64, strided<[1, 4]>>
!type_b = memref<1x2xf64, strided<[1, 1]>>
!type_c = memref<4x2xf64, strided<[1, 4]>>
!type_cast = memref<?x?xf64, strided<[1, ?]>>

func.func private @matmul(%a: !type_cast,
  %b: !type_cast,
  %c: !type_cast) {
  // linalg.matmul ins(%a, %b : !type_cast, !type_cast) outs(%c : !type_cast) -> ()
  %d0 = arith.constant 0 : index
  %d1 = arith.constant 1 : index
  %M = memref.dim %a, %d0 : !type_cast
  %K = memref.dim %a, %d1 : !type_cast
  %N = memref.dim %b, %d1 : !type_cast
  affine.for %m = 0 to %M {
    affine.for %n = 0 to %N {
      affine.for %k = 0 to %K {
        %a_v = affine.load %a[%m, %k] : !type_cast
        %b_v = affine.load %b[%k, %n] : !type_cast
        %c_v = affine.load %c[%m, %n] : !type_cast
        %prod = arith.mulf %a_v, %b_v : f64
        %res = arith.addf %prod, %c_v : f64
        affine.store %res, %c[%m, %n] : !type_cast
      }
    }
  }
  return
}


// func.func @wrap_matmul(%a: !type_a,
//   %b: !type_b,
//   %c: !type_c) {
//   %cast_a = memref.cast %a : !type_a to !type_cast
//   %cast_b = memref.cast %b : !type_b to !type_cast
//   %cast_c = memref.cast %c : !type_c to !type_cast
//   call @matmul(%cast_a, %cast_b, %cast_c) : (!type_cast, !type_cast, !type_cast) -> ()
//   return
// }


