module {

func.func @wrap (%ptr : memref<?xi32>, %idx : memref<?xi32>) {

    %data = quiccir.alloc_data(%ptr, %idx) : (memref<?xi32>, memref<?xi32>) -> memref<?xcomplex<f32>> {layout = "DCCSC3D"}
    // lowers to
    // %ptrStruct = builtin.unrealized_conversion_cast %ptr : llvm.ptr<struct<...>>
    // %idxStruct = builtin.unrealized_conversion_cast %idx : llvm.ptr<struct<...>>
    // %dataStruct = llvm.alloca %one x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>
    // func.call @alloc_data_complexf32_DCCSC3D(%dataStruct, %ptrStruct, %idxStruct)
    // %data = builtin.unrealized_conversion_cast %dataStruct : memref<?xcomplex<f32>>
    return
}

}

// ./bin/quiccir-opt ../examples/lower-alloc_data.mlir --convert-quiccir-to-llvm --lower-quiccir-alloc --convert-func-to-llvm