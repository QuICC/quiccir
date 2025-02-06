// RUN: quiccir-opt %s --quiccir-transform-group | FileCheck %s

module {
  // CHECK: func.func @entryGroupAll(%[[A0:.*]]: tensor<?x?x?xcomplex<f64>>, %[[A1:.*]]: tensor<?x?x?xcomplex<f64>>) -> tensor<?x?x?xcomplex<f64>> {
  // CHECK: %[[TRA:.*]]:2 = quiccir.transpose %[[A0]], %[[A1]] permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  // CHECK: return %[[TRA]]#0, %[[TRA]]#1 : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  func.func @entryGroupAll(%alphys0: tensor<?x?x?xcomplex<f64>>, %alphys1: tensor<?x?x?xcomplex<f64>>) -> tensor<?x?x?xcomplex<f64>> {
  %ftmod0 = quiccir.transpose %alphys0 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftmod1 = quiccir.transpose %alphys1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  return %ftmod0, %ftmod1  : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  }
}
