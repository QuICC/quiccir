%b0 = : tensor<3x2xf32>
%umod = : tensor<16x2x2xf32>
%partial = : tensor<16x3x2xf32>

// b0*umod
%result = linalg.generic {
  indexing_maps = [affine_map<(e, i, j, k) -> (i, k)>,
                   affine_map<(e, i, j, k) -> (e, k, j)>,
                   affine_map<(e, i, j, k) -> (e, i, j)>],
  iterator_types = ["parallel","parallel", "parallel", "reduction"]
} ins(%b0, %umod : tensor<3x2xf32>,tensor<16x2x2xf32>)
  outs(%partial :tensor<16x3x2xf32>) {
^bb0(%b0_one: f32, %umod_one: f32, %partial_one: f32):
  %0 = arith.mulf %b0_one, %umod_one : f32
  %1 = arith.addf %partial_one, %0 : f32
  linalg.yield %1 : f32
} -> tensor<16x3x2xf32>

