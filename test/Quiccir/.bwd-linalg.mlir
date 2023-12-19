// RUN: quiccir-opt %s --convert-quiccir-to-linalg | FileCheck %s

module {
    func.func @simple_no_arg() {
        %c0 = arith.constant 1.0 : f32
        %b0 = tensor.splat %c0 : tensor<3x2xf32>
        %b1 = tensor.splat %c0 : tensor<3x2xf32>
        %umod = tensor.splat %c0 : tensor<16x2x2xf32>
        // CHECK: %{{.*}} = tensor.empty() : tensor<16x3x2xf32>
        // CHECK: %{{.*}} = linalg.fill {{.*}} -> tensor<16x3x2xf32>
        // CHECK: %{{.*}} = linalg.generic {indexing_maps = [{{.*}}], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        // CHECK: } -> tensor<16x3x2xf32>
        // CHECK: %{{.*}} = tensor.empty() : tensor<2x3xf32>
        // CHECK: %{{.*}} = linalg.generic {indexing_maps = [{{.*}}], iterator_types = ["parallel", "parallel"]}
        // CHECK: } -> tensor<2x3xf32>
        // CHECK: %{{.*}} = tensor.empty() : tensor<16x3x3xf32>
        // CHECK: %{{.*}} = linalg.fill {{.*}} -> tensor<16x3x3xf32>
        // CHECK: %{{.*}} = linalg.generic {indexing_maps = [{{.*}}], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        // CHECK: } -> tensor<16x3x3xf32>
        %0 = quiccir.quadrature %b0, %b1, %umod : tensor<3x2xf32>, tensor<3x2xf32>, tensor<16x2x2xf32> -> tensor<16x3x3xf32>
        return
    }
}

// ./bin/quiccir-opt ../test/Quiccir/bwd-linalg.mlir --convert-quiccir-to-linalg
