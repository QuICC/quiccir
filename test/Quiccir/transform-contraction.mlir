// RUN: quiccir-opt %s --quiccir-transform-contraction | FileCheck %s

module {
  // CHECK: func.func @entry(%[[arg0:.*]]: tensor<?x?x?xcomplex<f64>>, %[[arg1:.*]]: tensor<?x?x?xcomplex<f64>>) -> tensor<?x?x?xf64> {
  // CHECK: %[[ADD:.*]] = quiccir.add %[[arg0]], %[[arg1]] : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  // CHECK: %[[TRA:.*]] = quiccir.transpose %[[ADD]] permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  // CHECK: %[[RET:.*]] = quiccir.fr.prj %[[TRA]] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes {kind = "P"}
  // CHECK: return %[[RET]] : tensor<?x?x?xf64>
  func.func @entryAddFr(%alphys0: tensor<?x?x?xcomplex<f64>>, %alphys1: tensor<?x?x?xcomplex<f64>>) -> tensor<?x?x?xf64> {
  %ftmod0 = quiccir.transpose %alphys0 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftphys0 = quiccir.fr.prj %ftmod0 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes {kind = "P"}
  %ftmod1 = quiccir.transpose %alphys1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftphys1 = quiccir.fr.prj %ftmod1 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes {kind = "P"}
  %add = quiccir.add %ftphys0, %ftphys1 : tensor<?x?x?xf64>, tensor<?x?x?xf64> -> tensor<?x?x?xf64>
  return %add : tensor<?x?x?xf64>
  }

  func.func @entryDifferentKind(%alphys0: tensor<?x?x?xcomplex<f64>>, %alphys1: tensor<?x?x?xcomplex<f64>>) -> tensor<?x?x?xf64> {
  %ftmod0 = quiccir.transpose %alphys0 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftphys0 = quiccir.fr.prj %ftmod0 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes {kind = "P"}
  %ftmod1 = quiccir.transpose %alphys1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftphys1 = quiccir.fr.prj %ftmod1 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes {kind = "D1"}
  %add = quiccir.add %ftphys0, %ftphys1 : tensor<?x?x?xf64>, tensor<?x?x?xf64> -> tensor<?x?x?xf64>
  return %add : tensor<?x?x?xf64>
  }

  // CHECK: func.func @entry(%[[arg0:.*]]: tensor<?x?x?xcomplex<f64>>, %[[arg1:.*]]: tensor<?x?x?xcomplex<f64>>) -> tensor<?x?x?xcomplex<f64>> {
  // CHECK: %[[SUB:.*]] = quiccir.sub %[[arg0]], %[[arg1]] : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  // CHECK: %[[TRA:.*]] = quiccir.transpose %[[SUB]] permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  // CHECK: %[[RET:.*]] = quiccir.al.prj %[[TRA]] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes {kind = "P"}
  // CHECK: return %[[RET]] : tensor<?x?x?xcomplex<f64>>
  func.func @entrySubAl(%alphys0: tensor<?x?x?xcomplex<f64>>, %alphys1: tensor<?x?x?xcomplex<f64>>) -> tensor<?x?x?xcomplex<f64>> {
  %ftmod0 = quiccir.transpose %alphys0 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftphys0 = quiccir.al.prj %ftmod0 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes {kind = "P"}
  %ftmod1 = quiccir.transpose %alphys1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftphys1 = quiccir.al.prj %ftmod1 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes {kind = "P"}
  %sub = quiccir.sub %ftphys0, %ftphys1 : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  return %sub : tensor<?x?x?xcomplex<f64>>
  }
}
