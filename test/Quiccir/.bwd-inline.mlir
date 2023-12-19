// RUN: quiccir-opt -inline %s | FileCheck %s

func.func private @simple(%b0: tensor<?x?xf32>, %b1: tensor<?x?xf32>, %umod: tensor<?x?x?xf32> ) -> (tensor<?x?x?xf32>) {
    %0 = quiccir.quadrature %b0, %b1, %umod : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32> -> tensor<?x?x?xf32>
    return %0 : tensor<?x?x?xf32>
}

func.func @wrapper(%b0: tensor<3x2xf32>, %b1: tensor<3x3xf32>, %umod: tensor<3x2x16xf32>) -> (tensor<3x3x16xf32>){
    %cast_b0 = tensor.cast %b0 : tensor<3x2xf32> to tensor<?x?xf32>
    %cast_b1 = tensor.cast %b1 : tensor<3x3xf32> to tensor<?x?xf32>
    %cast_umod = tensor.cast %umod : tensor<3x2x16xf32> to tensor<?x?x?xf32>
    // CHECK: %{{.*}} = quiccir.quadrature %{{.*}}, %{{.*}}, %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32> -> tensor<?x?x?xf32>
    %ret = call @simple(%cast_b0, %cast_b1, %cast_umod) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>)
    %cast_ret = tensor.cast %ret : tensor<?x?x?xf32> to tensor<3x3x16xf32>
    return %cast_ret : tensor<3x3x16xf32>
}
