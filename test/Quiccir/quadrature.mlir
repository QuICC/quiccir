// RUN: quiccir-opt -mlir-print-op-generic %s | FileCheck %s

module {
    func.func @simple(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32> ) {
        // CHECK: %{{.*}} = "quiccir.quadrature"(%{{.*}}, %{{.*}}) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        %0 = "quiccir.quadrature"(%arg0, %arg1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        return
    }
}
