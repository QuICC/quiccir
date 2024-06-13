module {

// memref.global "private" constant @__cst_ptr_4xi32 : memref<4xi32> = dense<[1, 2, 3, 4]>

func.func @main (%viewProd: !quiccir.view<16x2x3xcomplex<f32>, "layoutProd">) {
    %ptr = quiccir.pointers %viewProd : !quiccir.view<16x2x3xcomplex<f32>, "layoutProd"> -> memref<?xi32>
    %idx = quiccir.indices %viewProd : !quiccir.view<16x2x3xcomplex<f32>, "layoutProd"> -> memref<?xi32>
    %lds = llvm.mlir.constant(3 : i64) : i64
    %data = quiccir.alloc_data(%ptr, %idx), %lds : (memref<?xi32>, memref<?xi32>), i64 -> memref<?xcomplex<f32>> {layout = "layoutNew"}
    %view = quiccir.assemble(%ptr, %idx), %data : (memref<?xi32>, memref<?xi32>), memref<?xcomplex<f32>> -> !quiccir.view<16x3x3xcomplex<f32>, "layoutNew">
    return
}

}

// ./bin/quiccir-opt ../examples/quiccirToLLVM.mlir --convert-quiccir-to-llvm --lower-quiccir-alloc --convert-func-to-llvm --canonicalize --cse --finalize-quiccir-view --canonicalize