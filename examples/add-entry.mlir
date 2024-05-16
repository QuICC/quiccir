module {

func.func private @entry(%arg0: tensor<?x?x?xf64>, %arg1: tensor<?x?x?xf64>) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>)

}
// ./bin/quiccir-opt ../examples/add-entry.mlir --quiccir-view-wrapper='dim-rets=1,1,1 dim-args=2,2,2'