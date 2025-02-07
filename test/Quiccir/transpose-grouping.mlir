// RUN: quiccir-opt %s --quiccir-transpose-group | FileCheck %s
// RUN: quiccir-opt %s --quiccir-transpose-group='group=2' | FileCheck %s -check-prefix=PARTIAL

module {
  // CHECK: func.func @entryGroup2(%[[A0:.*]]: tensor<?x?x?xcomplex<f64>>, %[[A1:.*]]: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>) {
  // CHECK: %[[TRA:.*]]:2 = quiccir.transpose %[[A0]], %[[A1]] permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  // CHECK: return %[[TRA]]#0, %[[TRA]]#1 : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  func.func @entryGroup2(%alphys0: tensor<?x?x?xcomplex<f64>>, %alphys1: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>) {
  %ftmod0 = quiccir.transpose %alphys0 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftmod1 = quiccir.transpose %alphys1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  return %ftmod0, %ftmod1  : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  }

  // CHECK: func.func @entryGroup3(%[[A0:.*]]: tensor<?x?x?xcomplex<f64>>, %[[A1:.*]]: tensor<?x?x?xcomplex<f64>>, %[[A2:.*]]: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>) {
  // CHECK: %[[TRA:.*]]:3 = quiccir.transpose %[[A0]], %[[A1]], %[[A2]] permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  // CHECK: return %[[TRA]]#0, %[[TRA]]#1, %[[TRA]]#2 : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  // PARTIAL: func.func @entryGroup3(%[[A0:.*]]: tensor<?x?x?xcomplex<f64>>, %[[A1:.*]]: tensor<?x?x?xcomplex<f64>>, %[[A2:.*]]: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>) {
  // PARTIAL: %[[TRA:.*]]:2 = quiccir.transpose %[[A0]], %[[A1]] permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  // PARTIAL: %[[TRA2:.*]] = quiccir.transpose %[[A2]] permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  // PARTIAL: return %[[TRA]]#0, %[[TRA]]#1, %[[TRA2]] : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  func.func @entryGroup3(%alphys0: tensor<?x?x?xcomplex<f64>>, %alphys1: tensor<?x?x?xcomplex<f64>>, %alphys2: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>) {
  %ftmod0 = quiccir.transpose %alphys0 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftmod1 = quiccir.transpose %alphys1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftmod2 = quiccir.transpose %alphys2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  return %ftmod0, %ftmod1, %ftmod2  : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  }

  // CHECK: func.func @entryGroup4(%[[A0:.*]]: tensor<?x?x?xcomplex<f64>>, %[[A1:.*]]: tensor<?x?x?xcomplex<f64>>, %[[A2:.*]]: tensor<?x?x?xcomplex<f64>>, %[[A3:.*]]: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>) {
  // CHECK: %[[TRA:.*]]:4 = quiccir.transpose %[[A0]], %[[A1]], %[[A2]], %[[A3]] permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  // CHECK: return %[[TRA]]#0, %[[TRA]]#1, %[[TRA]]#2, %[[TRA]]#3 : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  // PARTIAL: func.func @entryGroup4(%[[A0:.*]]: tensor<?x?x?xcomplex<f64>>, %[[A1:.*]]: tensor<?x?x?xcomplex<f64>>, %[[A2:.*]]: tensor<?x?x?xcomplex<f64>>, %[[A3:.*]]: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>) {
  // PARTIAL: %[[TRA0:.*]]:2 = quiccir.transpose %[[A0]], %[[A1]] permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  // PARTIAL: %[[TRA1:.*]]:2 = quiccir.transpose %[[A2]], %[[A3]] permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  // PARTIAL: return %[[TRA0]]#0, %[[TRA0]]#1, %[[TRA1]]#0, %[[TRA1]]#1 : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  func.func @entryGroup4(%alphys0: tensor<?x?x?xcomplex<f64>>, %alphys1: tensor<?x?x?xcomplex<f64>>, %alphys2: tensor<?x?x?xcomplex<f64>>, %alphys3: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>) {
  %ftmod0 = quiccir.transpose %alphys0 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftmod1 = quiccir.transpose %alphys1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftmod2 = quiccir.transpose %alphys2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftmod3 = quiccir.transpose %alphys3 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  return %ftmod0, %ftmod1, %ftmod2, %ftmod3  : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
  }
}
