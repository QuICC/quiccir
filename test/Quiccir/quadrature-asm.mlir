// RUN: quiccir-opt %s | FileCheck %s

module {
    func.func @simple_no_arg() {
        %c0 = arith.constant 1.0 : f32
        %op = tensor.splat %c0 : tensor<16x3x2xf32>
        %umod = tensor.splat %c0 : tensor<16x2x2xf32>
        // CHECK: %{{.*}} = quiccir.quadrature %{{.*}}, %{{.*}} : tensor<16x3x2xf32>, tensor<16x2x2xf32> -> tensor<16x3x3xf32>
        %0 = quiccir.quadrature %op, %umod : tensor<16x3x2xf32>, tensor<16x2x2xf32> -> tensor<16x3x3xf32>
        return
    }
}
