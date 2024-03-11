// RUN: quiccir-opt -mlir-print-op-generic %s | FileCheck %s

module {
    func.func @wrap(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32> ) -> tensor<?x?x?xf32> {
        // CHECK: %{{.*}} = "quiccir.quadrature"(%{{.*}}, %{{.*}}) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        %0 = "quiccir.quadrature"(%arg0, %arg1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        return %0 : tensor<?x?x?xf32>
    }
}

module {
    func.func @wrap(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
        // CHECK: %{{.*}} = "quiccir.jw.prj"(%{{.*}}) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        %0 = "quiccir.jw.prj"(%arg0) : ( tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        return %0 : tensor<?x?x?xf32>
    }
}

module {
    func.func @wrap(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
        // CHECK: %{{.*}} = "quiccir.jw.prj"(%{{.*}}) {implptr = {{.*}} : i64} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        %0 = "quiccir.jw.prj"(%arg0) {implptr = 2 : i64}: ( tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        return %0 : tensor<?x?x?xf32>
    }
}
