// RUN: quiccir-opt %s --quiccir-set-implptr | FileCheck %s

module {
    func.func @entry(%jwmod: tensor<7x3x6xcomplex<f64>, "DCCSC3D_t">) {
    // CHECK: %[[JWP:.*]] = quiccir.jw.prj %{{.*}} : tensor<7x3x6xcomplex<f64>, "DCCSC3D_t"> -> tensor<7x6x6xcomplex<f64>, "DCCSC3D_t"> attributes {implptr = 0 : i64, kind = "P"}
    %jwphys = quiccir.jw.prj %jwmod : tensor<7x3x6xcomplex<f64>, "DCCSC3D_t"> -> tensor<7x6x6xcomplex<f64>, "DCCSC3D_t"> attributes {kind = "P"}
    // CHECK: %[[ALM:.*]] = quiccir.transpose %[[JWP]] permutation = [1, 2, 0] : tensor<7x6x6xcomplex<f64>, "DCCSC3D_t"> -> tensor<6x7x6xcomplex<f64>, "S1CLCSC3D_t"> attributes {implptr = 1 : i64}
    %almod = quiccir.transpose %jwphys permutation = [1, 2, 0] : tensor<7x6x6xcomplex<f64>, "DCCSC3D_t"> -> tensor<6x7x6xcomplex<f64>, "S1CLCSC3D_t">
    // CHECK: %[[ALP:.*]] = quiccir.al.prj %[[ALM]] : tensor<6x7x6xcomplex<f64>, "S1CLCSC3D_t"> -> tensor<6x10x6xcomplex<f64>, "DCCSC3D_t"> attributes {implptr = 2 : i64, kind = "P"}
    %alphys = quiccir.al.prj %almod : tensor<6x7x6xcomplex<f64>, "S1CLCSC3D_t"> -> tensor<6x10x6xcomplex<f64>, "DCCSC3D_t"> attributes {kind = "P"}
    // CHECK: %[[FTM:.*]] = quiccir.transpose %[[ALP]] permutation = [1, 2, 0] : tensor<6x10x6xcomplex<f64>, "DCCSC3D_t"> -> tensor<6x6x10xcomplex<f64>, "DCCSC3D_t"> attributes {implptr = 3 : i64}
    %ftmod = quiccir.transpose %alphys permutation = [1, 2, 0] : tensor<6x10x6xcomplex<f64>, "DCCSC3D_t"> -> tensor<6x6x10xcomplex<f64>, "DCCSC3D_t">
    // CHECk %{{.*}} = quiccir.fr.prj %[[FTM]] : tensor<6x6x10xcomplex<f64>, "DCCSC3D_t"> -> tensor<6x10x10xf64, "DCCSC3D_t"> attributes {implptr = 4 : i64, kind = "P"}
    %ftphys = quiccir.fr.prj %ftmod : tensor<6x6x10xcomplex<f64>, "DCCSC3D_t"> -> tensor<6x10x10xf64, "DCCSC3D_t"> attributes {kind = "P"}
    return
    }
}
