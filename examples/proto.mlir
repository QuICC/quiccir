%5 = quiccir.alloc(%viewProd) : !quiccir.view<6x6x10xcomplex<f64>, "DCCSC3D"> -> !quiccir.view<6x10x6xcomplex<f64>, "DCCSC3D"> {producer = "quiccir.transpose_201"}

// for transpose ops lowers to (or replaced by...)
// constant or load from input
%idx = ... : memref<?xi32>
%ptr = ... : memref<?xi32>
// external ?
// memref.global "private" @y : memref<4xi32>
// constant or load ptr
// ... build struct ...
// %idx = builtin.unrealized_conversion_cast %struct : !llvm.struct<...> to !memref<...>
// ...
// for other ops reuse producer metadata (need to lower ot LLVM)
%ptr = quiccir.pointers %viewProd : memref<?xi32>
%idx = quiccir.indices %viewProd : memref<?xi32>
// and (perhaps one can check the assemble op ret val and remove the type)
%data = quiccir.alloc_data(%idx, %ptr) : (memref<?xi32>, memref<?xi32>) -> memref<?xcomplex<f64>> {layout = "DCCSC3D"}
%5 = quiccir.assemble(%idx, %ptr), %data : (memref<?xi32>, memref<?xi32>), memref<?xcomplex<f64>> -> !quiccir.view<6x10x6xcomplex<f64>, "DCCSC3D">


... // lowers to
%c0 = arith.constant 0 : index
%idxSize = memref.dim %idx, %c0 : memref<?xi32>
%0 = memref.extract_aligned_pointer_as_index %idx : memref<?xi32> -> index
%1 = arith.index_cast %0 : index to i64
%idxPtr = llvm.inttoptr %1 : i64 to !llvm.ptr

%12 = llvm.mlir.undef : !llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>
%13 = llvm.insertvalue %3, %12[0, 2] : !llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>
%14 = llvm.insertvalue %2, %13[0, 0] : !llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>
%15 = llvm.insertvalue %3, %14[0, 1] : !llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>
// insert pointers to memref
...
%16 = llvm.alloca %1 x !llvm.struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)> : (i64) -> !llvm.ptr<struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>>
llvm.store %15, %16 : !llvm.ptr<struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>>
%17 = builtin.unrealized_conversion_cast %16 : !llvm.ptr<struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>> to !quiccir.view<6x10x6xcomplex<f64>, "DCCSC3D">
...
call @_ciface_quiccir_alloc_data_DCCSC3D(%17, %idxPtr, %idxSize, %ptrPtr, %ptrSize) : (!quiccir.view<6x10x6xcomplex<f64>, "DCCSC3D">, ptr<i32>, i32, ptr<i32>, ptr<i32>, i32, ptr<i32>>) -> ()

