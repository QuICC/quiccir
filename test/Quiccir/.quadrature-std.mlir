// RUN: quiccir-opt %s --convert-quiccir-to-call | FileCheck %s

module {
    func.func @simple_no_arg() -> tensor<16x3x2xf32> {
        %c0 = arith.constant 1.0 : f32
        %op = tensor.splat %c0 : tensor<16x3x1xf32>
        %umod = tensor.splat %c0 : tensor<16x1x2xf32>
        // CHECK: %{{.*}} = memref.alloc() : memref<16x3x2xf32>
        // CHECK: %{{.*}} = bufferization.to_memref %{{.*}} : memref<16x3x1xf32>
        // CHECK: %{{.*}} = bufferization.to_memref %{{.*}} : memref<16x1x2xf32>
        // CHECK: call @quiccir_quadrature(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<16x3x2xf32{{.*}}>, memref<16x3x1xf32{{.*}}>, memref<16x1x2xf32{{.*}}>) -> ()
        %0 = quiccir.quadrature %op, %umod : tensor<16x3x1xf32>, tensor<16x1x2xf32> -> tensor<16x3x2xf32>
        return %0 : tensor<16x3x2xf32>
    }
}