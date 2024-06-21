// RUN: quiccir-opt %s -canonicalize | FileCheck %s

module {
    func.func @wrap(%arg0: tensor<1x1x1xcomplex<f32>, "lay">) -> tensor<1x2x1xf32, "lay"> {
         %arg0t = tensor.cast %arg0 : tensor<1x1x1xcomplex<f32>, "lay"> to tensor<?x?x?xcomplex<f32>>
        // CHECK: %{{.*}} = quiccir.fr.prj %{{.*}} : tensor<1x1x1xcomplex<f32>, "lay"> -> tensor<1x2x1xf32, "lay"> attributes {implptr = 2 : i64}
        %0 = quiccir.fr.prj %arg0t : tensor<?x?x?xcomplex<f32>> -> tensor<?x?x?xf32> attributes {implptr = 2 : i64}
        %ret = tensor.cast %0 : tensor<?x?x?xf32> to tensor<1x2x1xf32, "lay">
        return %ret : tensor<1x2x1xf32, "lay">
    }

    func.func @wrapAdd(%arg0: tensor<1x1x1xf64, "lay">) -> tensor<1x1x1xf64, "lay"> {
        %arg0t = tensor.cast %arg0 : tensor<1x1x1xf64, "lay"> to tensor<?x?x?xf64>
        // CHECK: %{{.*}} = quiccir.add %{{.*}}, %{{.*}} : tensor<1x1x1xf64, "lay">, tensor<1x1x1xf64, "lay"> -> tensor<1x1x1xf64, "lay">
        %0 = quiccir.add %arg0t, %arg0t : tensor<?x?x?xf64>, tensor<?x?x?xf64> -> tensor<?x?x?xf64>
        %ret = tensor.cast %0 : tensor<?x?x?xf64> to tensor<1x1x1xf64, "lay">
        return %ret : tensor<1x1x1xf64, "lay">
    }

    func.func @wrapSub(%arg0: tensor<1x1x1xf64, "lay">) -> tensor<1x1x1xf64, "lay"> {
        %arg0t = tensor.cast %arg0 : tensor<1x1x1xf64, "lay"> to tensor<?x?x?xf64>
        // CHECK: %{{.*}} = quiccir.sub %{{.*}}, %{{.*}} : tensor<1x1x1xf64, "lay">, tensor<1x1x1xf64, "lay"> -> tensor<1x1x1xf64, "lay">
        %0 = quiccir.sub %arg0t, %arg0t : tensor<?x?x?xf64>, tensor<?x?x?xf64> -> tensor<?x?x?xf64>
        %ret = tensor.cast %0 : tensor<?x?x?xf64> to tensor<1x1x1xf64, "lay">
        return %ret : tensor<1x1x1xf64, "lay">
    }

}

