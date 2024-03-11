//
// Usage prototype, it is not meant to be run
//

//
// 1D example, 3 implementations options
//   1- gen c++ tree as now
//


//
// 1D example
//   2- implementation pointer pass
//     -> pointer pass
//     -> lower to lib call
//
module {
  // entry point
  func.func @entry(%implPtrs: !llvm.array<? x ptr>, %newumod: !quiccir.view<?x?x?xf64, "layuval">, %umod: !quiccir.view<?x?x?xf64, "layumod">) {
    // adaptor to tensor
    %tumod = quiccir.to_tensor %umod : !quiccir.view<?x?x?xf64, "layumod">
    // mods to phys transforms
    %uphys = quiccir.jw.prj %umods : tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes{implptr = 0 :i64, dim = ? :i64}
    // non linear step
    %tnlin = quiccir.add %uphys, %uphys : (tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> tensor<?x?x?xf64> attributes{implptr = 1 :i64, dim = ? :i64}
    // phys to mods transforms
    %tnewmod = quiccir.jw.int %umods : tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes{implptr = 2 :i64, dim = ? :i64}
    // adaptor to view
    %newumod = quiccir.to_view %tnewmod : !quiccir.view<?x?x?xf64, "layuval">
  }
}

module {
  // entry point
  func.func @entry(%implPtrs: !llvm.array<? x ptr>, %newumod: !quiccir.view<?x?x?xf64, "layuval">, %umod: !quiccir.view<?x?x?xf64, "layumod">) {
    // adaptor to tensor
    // removed with canonicalize
    %tumod = quiccir.to_tensor %umod : !quiccir.view<?x?x?xf64, "layumod">
    // or temporarily if we don't care about codegen and checking size/attribute
    // %tumod = builtin.unrealized_conversion_cast %umod : !type_view to tensor<?x?x?xf64, "layumod">
    // mods to phys transforms
    %this_0 = llvm.extractvalue %implPtrs [0] : !llvm.array<?x ptr>
    // alloc return value // lowert to call to pool
    %uphys = quiccir.alloc() : !quiccir.view<?x?x?xf64, "layout depends on %umod and quiccir.jw.prj">
    call @quiccir_jw_prj(%this_0, %umod, %uphys) : () -> (!llvm.ptr, !quiccir.view<?x?x?xf64>, !quiccir.view<?x?x?xf64>)
    // add dealloc to pool
    // ...
  }
}


module {
  // entry point
  func.func @entry(%implPtrs: !llvm.array<? x ptr>, %newumod: !quiccir.view<?x?x?xf64>, %umod: !quiccir.view<?x?x?xf64>) {
    // adaptor to tensor
    // remove with canonicalize
    %tumod = quiccir.to_tensor %umod : !quiccir.view<?x?x?xf64>
    // or temporarily if we don't care about codegen
    // %tumod = builtin.unrealized_conversion_cast %umod : !type_view to tensor<?x?x?xf64>
    // mods to phys transforms
    %this_0 = llvm.extractvalue %implPtrs [0] : !llvm.array<?x ptr>
    // alloc return value
    %uphys_pos = memref.alloc() : memref<?xi32>
    %uphys_coo = memref.alloc() : memref<?xi32>
    %uphys_val = memref.alloc() : memref<?xf64>
    %uphys = quiccir.assemble(%uphys_pos, %uphys_coo, %uphys_val) : !quiccir.view<?x?x?xf64>
    call @quiccir_jw_prj(%this_0, %umod, %uphys) : () -> (!llvm.ptr, !quiccir.view<?x?x?xf64>, !quiccir.view<?x?x?xf64>)
    // ...
  }
}
// then
// quiccir.view -> llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>
//
// alternatively one could pass an array of buffers
//
module {
  // entry point
  func.func @entry(%implPtrs: !llvm.array<? x ptr>, %bufPtr: !llvm.array<? x ptr>, %newumod: !quiccir.view<?x?x?xf64>, %umod: !quiccir.view<?x?x?xf64>) {
    // adaptor to tensor
    // remove with canonicalize
    %tumod = quiccir.to_tensor %umod : !quiccir.view<?x?x?xf64>
    // mods to phys transforms
    %this_0 = llvm.extractvalue %implPtrs [0] : !llvm.array<?x ptr>
    // get return value from buffer
    // how do we map? use op index?
    %bprt = llvm.extractvalue %bufPtr [0] : !llvm.array<?x ptr>
    %buphys = llvm.load %bprt : !llvm.ptr -> !llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>
    %uphys = builtin.unrealized_conversion_cast %buphys : !llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)> to !quiccir.view<?x?x?xf64>
    call @quiccir_jw_prj(%this_0, %umod, %uphys) : () -> (!llvm.ptr, !quiccir.view<?x?x?xf64>, !quiccir.view<?x?x?xf64>)
    // ...
  }
}

//
// 1D example
//   3- codegen in MLIR
//     - pointer pass
//     - lower operator tensor init and batched_quad
//     - lower init to lib call
//     - lower batched quad to linalg
//     - codegen op in MLIR
//
module {
  // entry point
  func.func @entry(%implPtrs: !llvm.array<? x ptr>, %newumod: memref<?x?x?xf64>, %umod: memref<?x?x?xf64>) {
    // adaptor to tensor
    %tumod = bufferization.to_tensor %umod : memref<?x?x?xf64>
    // mods to phys transforms
    %uphys = quiccir.jw.prj %umods : tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes{implptr = 0 :i64, dim = ? :i64}
    // non linear step
    %tnlin = quiccir.add %uphys, %uphys : (tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> tensor<?x?x?xf64> attributes{implptr = 1 :i64, dim = ? :i64}
    // phys to mods transforms
    %tnewmod = quiccir.jw.int %umods : tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes{implptr = 2 :i64, dim = ? :i64}
    // adaptor to memref
    %newumod = bufferization.to_memref %tnewmod : memref<?x?x?xf64>
  }
}

module {
  // entry point
  func.func @entry(%implPtrs: !llvm.array<? x ptr>, %newumod: memref<?x?x?xf64>, %umod: memref<?x?x?xf64>) {
    // adaptor to tensor
    %tumod = bufferization.to_tensor %umod : memref<?x?x?xf64>
    // mods to phys transforms
    %top = quiccir.init : () -> tensor<?x?x?xf64> attributes{implptr = 0 :i64, op = "quiccir_jw_prj"}
    %uphys = quiccir.batched_quadrature %top, %tumod : (tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
    // ...
  }
}

module {
  // entry point
  func.func @entry(%implPtrs: !llvm.array<? x ptr>, %newumod: memref<?x?x?xf64>, %umod: memref<?x?x?xf64>) {
    // adaptor to tensor
    %tumod = bufferization.to_tensor %umod : memref<?x?x?xf64>
    // mods to phys transforms
    %this_0 = llvm.extractvalue %implPtrs [0] : !llvm.array<?x ptr>
    // ...
    %op = memref.alloc() : memref<?x?x?xf64> // this could be an external buffer to avoid realloc
    call @quiccir_jw_prj_get_op(%this_0, %op) : () -> (!llvm.ptr, memref<?x?x?xf64>)
    %top = bufferization.to_tensor %umod : memref<?x?x?xf64>
    %uphys = quiccir.batched_quadrature %top, %tumod : (tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
    // ...
  }
}