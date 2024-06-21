// RUN: quiccir-opt %s --set-quiccir-view-lay | FileCheck %s

func.func @entry(%arg0: !llvm.ptr<array<5 x ptr>> {llvm.noalias}, %arg1: !quiccir.view<5x2x3xcomplex<f64>, "layMMM">, %arg2: !quiccir.view<4x10x6xf64, "layPPP">) {
  %0 = builtin.unrealized_conversion_cast %arg2 : !quiccir.view<4x10x6xf64, "layPPP"> to tensor<4x10x6xf64>
  // CHECK: %{{.*}}= quiccir.fr.int %{{.*}} : tensor<4x10x6xf64, "layPPP"> -> tensor<4x3x6xcomplex<f64>, "layMPP">
  %1 = quiccir.fr.int %0 : tensor<4x10x6xf64> -> tensor<4x3x6xcomplex<f64>> attributes {implptr = 0 : i64}
  %2 = quiccir.transpose %1 permutation = [2, 0, 1] : tensor<4x3x6xcomplex<f64>> -> tensor<3x6x4xcomplex<f64>> attributes {implptr = 1 : i64}
  // CHECK: %{{.*}}= quiccir.al.int %{{.*}} : tensor<3x6x4xcomplex<f64>, "layPMP"> -> tensor<3x5x4xcomplex<f64>, "layMMP">
  %3 = quiccir.al.int %2 : tensor<3x6x4xcomplex<f64>> -> tensor<3x5x4xcomplex<f64>> attributes {implptr = 2 : i64}
  %4 = quiccir.transpose %3 permutation = [2, 0, 1] : tensor<3x5x4xcomplex<f64>> -> tensor<5x4x3xcomplex<f64>> attributes {implptr = 3 : i64}
  // CHECK: %{{.*}} = quiccir.jw.int %{{.*}} : tensor<5x4x3xcomplex<f64>, "layPMM"> -> tensor<5x2x3xcomplex<f64>, "layMMM">
  %5 = quiccir.jw.int %4 : tensor<5x4x3xcomplex<f64>> -> tensor<5x2x3xcomplex<f64>, "layMMM"> attributes {implptr = 4 : i64}
  quiccir.materialize %5 in %arg1 : (tensor<5x2x3xcomplex<f64>, "layMMM">, !quiccir.view<5x2x3xcomplex<f64>, "layMMM">)
  return
}

func.func @entryloop(%arg0: !llvm.ptr<array<10 x ptr>> {llvm.noalias}, %arg1: !quiccir.view<5x2x3xcomplex<f64>, "layMMM">, %arg2: !quiccir.view<5x2x3xcomplex<f64>, "layMMM">) {
  %0 = builtin.unrealized_conversion_cast %arg2 : !quiccir.view<5x2x3xcomplex<f64>, "layMMM"> to tensor<5x2x3xcomplex<f64>>
  // CHECK: %{{.*}}= quiccir.jw.prj %{{.*}} : tensor<5x2x3xcomplex<f64>, "layMMM"> -> tensor<5x4x3xcomplex<f64>, "layPMM">
  %1 = quiccir.jw.prj %0 : tensor<5x2x3xcomplex<f64>> -> tensor<5x4x3xcomplex<f64>> attributes {implptr = 0 : i64}
  %2 = quiccir.transpose %1 permutation = [1, 2, 0] : tensor<5x4x3xcomplex<f64>> -> tensor<3x5x4xcomplex<f64>> attributes {implptr = 1 : i64}
  // CHECK: %{{.*}}= quiccir.al.prj %{{.*}} : tensor<3x5x4xcomplex<f64>, "layMMP"> -> tensor<3x6x4xcomplex<f64>, "layPMP">
  %3 = quiccir.al.prj %2 : tensor<3x5x4xcomplex<f64>> -> tensor<3x6x4xcomplex<f64>> attributes {implptr = 2 : i64}
  %4 = quiccir.transpose %3 permutation = [1, 2, 0] : tensor<3x6x4xcomplex<f64>> -> tensor<4x3x6xcomplex<f64>> attributes {implptr = 3 : i64}
  // CHECK: %{{.*}}= quiccir.fr.prj %{{.*}} : tensor<4x3x6xcomplex<f64>, "layMPP"> -> tensor<4x10x6xf64, "layPPP">
  %5 = quiccir.fr.prj %4 : tensor<4x3x6xcomplex<f64>> -> tensor<4x10x6xf64> attributes {implptr = 4 : i64}
  // CHECK: %{{.*}}= quiccir.fr.int %{{.*}} : tensor<4x10x6xf64, "layPPP"> -> tensor<4x3x6xcomplex<f64>, "layMPP">
  %6 = quiccir.fr.int %5 : tensor<4x10x6xf64> -> tensor<4x3x6xcomplex<f64>> attributes {implptr = 5 : i64}
  %7 = quiccir.transpose %6 permutation = [2, 0, 1] : tensor<4x3x6xcomplex<f64>> -> tensor<3x6x4xcomplex<f64>> attributes {implptr = 6 : i64}
  // CHECK: %{{.*}}= quiccir.al.int %{{.*}} : tensor<3x6x4xcomplex<f64>, "layPMP"> -> tensor<3x5x4xcomplex<f64>, "layMMP">
  %8 = quiccir.al.int %7 : tensor<3x6x4xcomplex<f64>> -> tensor<3x5x4xcomplex<f64>> attributes {implptr = 7 : i64}
  %9 = quiccir.transpose %8 permutation = [2, 0, 1] : tensor<3x5x4xcomplex<f64>> -> tensor<5x4x3xcomplex<f64>> attributes {implptr = 8 : i64}
  // CHECK: %{{.*}} = quiccir.jw.int %{{.*}} : tensor<5x4x3xcomplex<f64>, "layPMM"> -> tensor<5x2x3xcomplex<f64>, "layMMM">
  %10 = quiccir.jw.int %9 : tensor<5x4x3xcomplex<f64>> -> tensor<5x2x3xcomplex<f64>> attributes {implptr = 9 : i64}
  quiccir.materialize %10 in %arg1 : (tensor<5x2x3xcomplex<f64>>, !quiccir.view<5x2x3xcomplex<f64>, "layMMM">)
  return
}


