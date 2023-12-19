// RUN: quiccir-opt %s | FileCheck %s

// // CHECK-LABEL: func @simple_er1
// func.func @simple_er1(%b0: tensor<1x3x4xf32>, %b1: tensor<3x4xf32>, %umod: tensor<4x4xf32> ) {
//     %0 = quiccir.quadrature %b0, %b1, %umod : tensor<1x3x4xf32>, tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x3xf32>
//     return
// }

// // CHECK-LABEL: func @simple_er2
// func.func @simple_er2(%b0: tensor<3x4xf32>, %b1: tensor<1x3x4xf32>, %umod: tensor<4x4xf32> ) {
//     %0 = quiccir.quadrature %b0, %b1, %umod : tensor<3x4xf32>, tensor<1x3x4xf32>, tensor<4x4xf32> -> tensor<3x3xf32>
//     return
// }

module {
    // CHECK-LABEL: func.func @simple
    func.func @simple(%b0: tensor<3x3xf32>, %b1: tensor<3x3xf32>, %umod: tensor<3x3xf32> ) {
        %0 = quiccir.quadrature %b0, %b1, %umod : tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32> -> tensor<3x3xf32>
        return
    }
}
