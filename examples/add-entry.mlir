module {

func.func private @tensor_entry(%arg0: tensor<?x?x?xf64, "lay">, %arg1: tensor<?x?x?xf64, "lay">) -> (tensor<?x?x?xf64, "lay">, tensor<?x?x?xf64, "lay">)

// func.func private @tensor_entry(%arg0: tensor<2x2x2xf64, "lay">, %arg1: tensor<2x2x2xf64, "lay">) -> (tensor<1x1x1xf64, "lay">, tensor<1x1x1xf64, "lay">)

}
// ./bin/quiccir-opt ../examples/add-entry.mlir --quiccir-view-wrapper


// module {

// func.func private @tensor_entry(%Polur: tensor<2x2x2xf64, "lay">) -> (tensor<1x1x1xf64, "lay">)

// }