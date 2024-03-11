//
// Usage prototype, it is not meant to be run
//

module {
  // entry point
  func.func @entry(%newumod: memref<?x?x?xf64>, %umod: memref<?x?x?xf64>) {
    // adaptor to tensor
    %tumod = bufferization.to_tensor %umod : memref<?x?x?xf64>
    // mods to phys transforms
    %tuval = call @mod2phys(%tumod) : tensor<?x?x?xf64> -> tensor<?x?x?xf64>
    // non linear step
    %tnlin = call @nlinOps(%tuval) : tensor<?x?x?xf64> -> tensor<?x?x?xf64>
    // phys to mods transforms
    %tnewmod = call @phys2mods(%tnlin) : tensor<?x?x?xf64> -> tensor<?x?x?xf64>
    // adaptor to memref
    %newumod = bufferization.to_memref %tnewmod : memref<?x?x?xf64>
  }

  // mods -> phys
  func.func private @mod2phys(%in: tensor<?x?x?xf64>) -> !tensor<?x?x?xf64>
  // nonlinear
  func.func private @nlinOps(%in: tensor<?x?x?xf64>) -> !tensor<?x?x?xf64>
  // phys -> mods
  func.func private @phys2mods(%in: tensor<?x?x?xf64>) -> !tensor<?x?x?xf64>
}

//
// 1D example
// after inlining
//
module {
  // entry point
  func.func @entry(%newumod: memref<?x?x?xf64>, %umod: memref<?x?x?xf64>) {
    // adaptor to tensor
    %tumod = bufferization.to_tensor %umod : memref<?x?x?xf64>
    // mods to phys transforms
    %uphys = quiccir.jw.prj %umods : tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes{dim = ? :i64}
    // non linear step
    %tnlin = quiccir.add %uphys, %uphys : (tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
    // phys to mods transforms
    %tnewmod = quiccir.jw.int %umods : tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes{dim = ? :i64}
    // adaptor to memref
    %newumod = bufferization.to_memref %tnewmod : memref<?x?x?xf64>
  }
}

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
    %this_0 = llvm.extractvalue %implPtrs [0] : !llvm.array<?x ptr>
    // ...
    %uphys = memref.alloc() : memref<?x?x?xf64>
    call @quiccir_jw_prj(%this_0, %umod, %uphys) : () -> (!llvm.ptr, memref<?x?x?xf64>, memref<?x?x?xf64>)
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