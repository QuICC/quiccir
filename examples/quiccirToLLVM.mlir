module {

// memref.global "private" constant @__cst_ptr_4xi32 : memref<4xi32> = dense<[1, 2, 3, 4]>

func.func @main (%viewProd: !quiccir.view<16x2x3xcomplex<f32>, "layoutProd">) {
    %ptr = quiccir.pointers %viewProd : !quiccir.view<16x2x3xcomplex<f32>, "layoutProd"> -> memref<?xi32>
    // %idx = quiccir.indices %viewProd : !quiccir.view<16x2x3xcomplex<f32>, "layoutProd"> -> memref<?xi32>
    // %data = quiccir.alloc_data(%ptr, %idx) : (memref<?xi32>, memref<?xi32>) -> memref<?xcomplex<f32>> {layout = "layoutNew"}
    // %view = quiccir.assemble(%ptr, %idx), %data : (memref<?xi32>, memref<?xi32>), memref<?xcomplex<f32>> -> !quiccir.view<16x3x3xcomplex<f32>, "layoutNew">
    return
}

}

// ./bin/quiccir-opt ../examples/quiccirToLLVM.mlir --convert-quiccir-to-llvm