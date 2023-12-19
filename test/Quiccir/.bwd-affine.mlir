// RUN: quiccir-opt %s --convert-quiccir-to-affine | FileCheck %s

module {
    func.func @simple_no_arg() {
        %c0 = arith.constant 1.0 : f32
        %b0 = tensor.splat %c0 : tensor<3x2xf32>
        %b1 = tensor.splat %c0 : tensor<3x2xf32>
        %umod = tensor.splat %c0 : tensor<16x2x2xf32>
        // CHECK: %{{.*}} = memref.alloc() : memref<16x3x2xf32>
        // CHECK: %{{.*}} = memref.alloc() : memref<16x3x3xf32>
        // CHECK: affine.for %arg0 = 0 to 16
        // CHECK: affine.for %arg1 = 0 to 3
        // CHECK: affine.for %arg2 = 0 to 2
        // CHECK: affine.for %arg3 = 0 to 2
        // CHECK: affine.for %arg0 = 0 to 16
        // CHECK: affine.for %arg1 = 0 to 3
        // CHECK: affine.for %arg2 = 0 to 3
        // CHECK: affine.for %arg3 = 0 to 2
        %0 = quiccir.quadrature %b0, %b1, %umod : tensor<3x2xf32>, tensor<3x2xf32>, tensor<16x2x2xf32> -> tensor<16x3x3xf32>
        return
    }
}

// ./bin/quiccir-opt ../test/Quiccir/bwd-affine.mlir --convert-quiccir-to-affine
