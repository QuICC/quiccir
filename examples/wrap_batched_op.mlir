!type_op = memref<1x3x2xf64>
!type_umod = memref<1x2x3xf64>
!type_uval = memref<1x3x3xf64>

!type_top = tensor<1x3x2xf64>
!type_tumod = tensor<1x2x3xf64>
!type_tuval = tensor<1x3x3xf64>
!type_tcast = tensor<?x?x?xf64>

func.func private @batched_matmul(%a: !type_tcast, %b: !type_tcast) -> !type_tcast {
  %dim0 = arith.constant 0 : index
  %dim1 = arith.constant 1 : index
  %dim2 = arith.constant 2 : index
  %E = tensor.dim %a, %dim0 : !type_tcast
  %M = tensor.dim %a, %dim1 : !type_tcast
  %N = tensor.dim %b, %dim2 : !type_tcast
  %init = tensor.empty(%E, %M, %N) : !type_tcast
  %result = linalg.generic {
    indexing_maps = [affine_map<(e, i, j, k) -> (e, i, k)>,
                     affine_map<(e, i, j, k) -> (e, k, j)>,
                     affine_map<(e, i, j, k) -> (e, i, j)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } ins(%a, %b : !type_tcast, !type_tcast)
    outs(%init : !type_tcast) {
  ^bb0(%a_one: f64, %b_one: f64, %init_one: f64):
    %mul = arith.mulf %a_one, %b_one : f64
    %sum = arith.addf %mul, %init_one : f64
    linalg.yield %sum : f64
  } -> !type_tcast
  return %result : !type_tcast
}

// mod -> phys
func.func @wrap_batched_op(%uval: !type_uval, %op: !type_op, %umod: !type_umod)
  attributes {llvm.emit_c_interface} {
  %top = bufferization.to_tensor %op : !type_op
  %tumod = bufferization.to_tensor %umod : !type_umod
  %cast_top = tensor.cast %top : !type_top to !type_tcast
  %cast_tumod = tensor.cast %tumod : !type_tumod to !type_tcast
  %ret = call @batched_matmul(%cast_top, %cast_tumod) : (!type_tcast, !type_tcast) -> (!type_tcast)
  %cast_ret = tensor.cast %ret : !type_tcast to !type_tuval
  %tmpref = bufferization.to_memref %cast_ret : !type_uval
  memref.copy %tmpref, %uval : !type_uval to !type_uval
  return
}

// ./bin/quiccir-opt ../examples/wrap_batched_op.mlir --inline --empty-tensor-to-alloc-tensor --linalg-bufferize --cse --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-cf-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm -cse --reconcile-unrealized-casts | ./bin/quiccir-miniapp -emit=jit