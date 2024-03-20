// RUN: quiccir-opt %s -canonicalize | FileCheck %s

module {
    func.func @wrap(%arg0: tensor<1x1x1xf32>) -> tensor<1x1x1xf32> {
        %0 = quiccir.fr.prj %arg0 : tensor<1x1x1xf32> -> tensor<?x?x?xf32> attributes {implptr = 2 : i64}
        %ret = tensor.cast %0 : tensor<?x?x?xf32> to tensor<1x1x1xf32>
        return %ret : tensor<1x1x1xf32>
    }

    // func.func @wrap(%arg0: tensor<1x1x1xf32>) -> tensor<1x1x1xf32> {
    //      %arg0t = tensor.cast %arg0 : tensor<1x1x1xf32> to tensor<?x?x?xf32>
    //     // CHECK: %{{.*}} = quiccir.fr.prj %{{.*}} : tensor<?x?x?xf32> -> tensor<?x?x?xf32> attributes {implptr = {{.*}} : i64}
    //     %0 = quiccir.fr.prj %arg0t : tensor<?x?x?xf32> -> tensor<?x?x?xf32> attributes {implptr = 2 : i64}
    //     %ret = tensor.cast %0 : tensor<?x?x?xf32> to tensor<1x1x1xf32>
    //     return %ret : tensor<1x1x1xf32>
    // }

    // func.func @wrapT(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1xf32>) -> tensor<1x1xf32> {
    //     %arg0t = tensor.cast %arg0 : tensor<1x1xf32> to tensor<?x?xf32>
    //     %arg1t = tensor.cast %arg1 : tensor<1x1xf32> to tensor<?x?xf32>
    //     %arg2t = tensor.cast %arg2 : tensor<1x1xf32> to tensor<?x?xf32>
    //     // CHECK: %{{.*}} = quiccir.fr.prj %{{.*}} : tensor<?x?xf32> -> tensor<?x?xf32> attributes {implptr = {{.*}} : i64}
    //     %rett = linalg.matmul ins(%arg1t, %arg2t : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg0t: tensor<?x?xf32>) -> tensor<?x?xf32>
    //     // %rett = linalg.add ins(%arg1t, %arg2t : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg0t: tensor<?x?xf32>) -> tensor<?x?xf32>
    //     %ret = tensor.cast %rett : tensor<?x?xf32> to tensor<1x1xf32>
    //     return %ret : tensor<1x1xf32>
    // }
}

