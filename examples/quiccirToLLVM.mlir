module {

// memref.global "private" constant @__cst_ptr_4xi32 : memref<4xi32> = dense<[1, 2, 3, 4]>

func.func @main (%arg: !quiccir.view<16x2x3xcomplex<f32>, "layoutUmod">) {

    %1 = quiccir.pointers %arg : !quiccir.view<16x2x3xcomplex<f32>, "layoutUmod"> -> memref<?xi32>
    // %10 = builtin.unrealized_conversion_cast %arg : !quiccir.view<16x2x3xcomplex<f32> to !llvm.struct<...>
    // %11 = builtin.unrealized_conversion_cast %arg : memref<?xi32> to !llvm.struct<...>
    // copy relevant bit from view struct to memref struct
    // %1 = builtin.unrealized_conversion_cast !llvm.struct<> to memref<?xi32>

    // %2 = quiccir.indices %arg : !quiccir.view<16x2x3xcomplex<f32>, "layoutUmod"> -> memref<?xi32>
    // // same as above

    // %3 = quiccir.alloc_data(%1, %2) : (memref<?xi32>, memref<?xi32>) -> memref<?xcomplex<f32>> {layout = "layoutUval"}
    // // lower to call

    // %4 = quiccir.assemble(%1, %2), %3 : (memref<?xi32>, memref<?xi32>), memref<?xcomplex<f32>> -> !quiccir.view<16x3x3xcomplex<f32>, "layoutUval">
    // // %10 = builtin.unrealized_conversion_cast %arg : !quiccir.view<16x2x3xcomplex<f32> to !llvm.struct<...>
    // // copy relevant bit from view struct to memref struct

    return
}


}

// mlir-opt ../examples/new-alloc.mlir --finalize-memref-to-llvm --convert-func-to-llvm -reconcile-unrealized-casts | mlir-cpu-runner -e main -entry-point-result=void --shared-libs=$MLIR_ROOT/lib/libmlir_c_runner_utils.so --shared-libs=$MLIR_ROOT/lib/libmlir_runner_utils.so


// %0 = llvm.mlir.constant(4 : index) : i64
// %1 = llvm.mlir.constant(1 : index) : i64
// %2 = llvm.mlir.null : !llvm.ptr
// %3 = llvm.getelementptr %2[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
// %5 = llvm.mlir.addressof @__cst_ptr_4xi32 : !llvm.ptr
// %6 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
// %7 = llvm.mlir.constant(3735928559 : index) : i64
// %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
// %9 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// %10 = llvm.insertvalue %8, %9[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// %11 = llvm.insertvalue %6, %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// %12 = llvm.mlir.constant(0 : index) : i64
// %13 = llvm.insertvalue %12, %11[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// %14 = llvm.insertvalue %0, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// %15 = llvm.insertvalue %1, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// %16 = builtin.unrealized_conversion_cast %15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<4xi32>
