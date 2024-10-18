// RUN: quiccir-opt %s --quiccir-transform-contraction | FileCheck %s

// Check enumeration and different type different op
module {
  func.func @entry(%alphys0: tensor<?x?x?xcomplex<f64>>, %alphys1: tensor<?x?x?xcomplex<f64>>) -> tensor<?x?x?xf64> {
  %ftmod0 = quiccir.transpose %alphys0 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftphys0 = quiccir.fr.prj %ftmod0 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes {kind = "P"}
  %ftmod1 = quiccir.transpose %alphys1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
  %ftphys1 = quiccir.fr.prj %ftmod1 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes {kind = "P"}
  %add = quiccir.add %ftphys0, %ftphys1 : tensor<?x?x?xf64>, tensor<?x?x?xf64> -> tensor<?x?x?xf64>
  return %add : tensor<?x?x?xf64>
  }
}

// module {
//   func.func @entry(%alphys0: tensor<?x?x?xcomplex<f64>>, %alphys1: tensor<?x?x?xcomplex<f64>>) -> tensor<?x?x?xf64> {
//   %add = quiccir.add %alphys0, %alphys1 : tensor<?x?x?xf64>, tensor<?x?x?xf64> -> tensor<?x?x?xf64>
//   %ftmod = quiccir.transpose %add permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
//   %ftphys = quiccir.fr.prj %ftmod : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes {kind = "P"}
//   return %ftphys : tensor<?x?x?xf64>
//   }
// }

