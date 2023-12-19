#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (8)>
#map3 = affine_map<() -> (0, 0)>
#map4 = affine_map<() -> (16, 8)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<16x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<16x8xf32>):
    "affine.parallel"() ({
    ^bb0(%arg3: index, %arg4: index):
      "affine.for"() ({
      ^bb0(%arg5: index):
        %0 = "affine.load"(%arg0, %arg3, %arg5) {map = #map0} : (memref<16x8xf32>, index, index) -> f32
        %1 = "affine.load"(%arg1, %arg5, %arg4) {map = #map0} : (memref<8x8xf32>, index, index) -> f32
        %2 = "affine.load"(%arg2, %arg3, %arg4) {map = #map0} : (memref<16x8xf32>, index, index) -> f32
        %3 = "arith.mulf"(%0, %1) : (f32, f32) -> f32
        %4 = "arith.addf"(%3, %2) : (f32, f32) -> f32
        "affine.store"(%4, %arg2, %arg3, %arg4) {map = #map0} : (f32, memref<16x8xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map3, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<16x8xf32>, memref<8x8xf32>, memref<16x8xf32>) -> (), sym_name = "matmul", sym_visibility = "private"} : () -> ()
}) : () -> ()
