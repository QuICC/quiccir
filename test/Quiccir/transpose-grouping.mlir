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

  // Here we check that we are NOT grouping the transposes
  //
  // CHECK: func.func @entryNoGroup(%[[A0:.*]]: tensor<?x?x?xcomplex<f64>>) -> tensor<?x?x?xf64> {
  func.func @entryNoGroup(%arg0: tensor<?x?x?xcomplex<f64>>) -> tensor<?x?x?xf64> {
    // CHECK: %[[JW:.*]] = quiccir.jw.prj %[[A0]] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes {kind = "P"}
    // CHECK: %[[TRAJW:.*]] = quiccir.transpose %[[JW]] permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
    // CHECK: %[[AL:.*]] = quiccir.al.prj %[[TRAJW]] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes {kind = "P"}
    // CHECK: %[[TRAAL:.*]] = quiccir.transpose %[[AL]] permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
    %0 = quiccir.jw.prj %arg0 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes {kind = "P"}
    %1 = quiccir.transpose %0 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
    %2 = quiccir.al.prj %1 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes {kind = "P"}
    %3 = quiccir.transpose %2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
    %4 = quiccir.fr.prj %3 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes {kind = "P"}
    return %4 : tensor<?x?x?xf64>
  }

  // Here we check the reordering of the transposes results uses
  //
  // CHECK: func.func @entryGroupReorder(%[[S:.*]]: tensor<?x?x?xcomplex<f64>>, %[[T:.*]]: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>) {
  func.func @entryGroupReorder(%S: tensor<?x?x?xcomplex<f64>>, %T: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>) {
      // CHECK: %[[S1:.*]] = quiccir.jw.prj %[[S]] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes {kind = "D1"}
      // CHECK: %[[T1:.*]] = quiccir.jw.prj %[[T]] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes {kind = "P"}
      // CHECK: %[[ST1T:.*]]:2 = quiccir.transpose %[[S1]], %[[T1]] permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
      // CHECK: %[[S2:.*]] = quiccir.al.prj %[[ST1T]]#0 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes {kind = "P"}
      // CHECK: %[[T2:.*]] = quiccir.al.prj %[[ST1T]]#1 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes {kind = "P"}
      // CHECK: %[[S2T:.*]]:2 = quiccir.transpose %[[S2]], %[[T2]] permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
      // CHECK: %[[S3:.*]] = quiccir.fr.prj %[[S2T]]#0 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes {kind = "P"}
      // CHECK: %[[T3:.*]] = quiccir.fr.prj %[[S2T]]#1 : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes {kind = "D1"}
      // CHECK: return %[[S3]], %[[T3]] : tensor<?x?x?xf64>, tensor<?x?x?xf64>
      %S1 = quiccir.jw.prj %S : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{kind = "D1"}
      %S1T = quiccir.transpose %S1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
      %S2 = quiccir.al.prj %S1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{kind = "P"}
      %S2T = quiccir.transpose %S2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
      %S3 = quiccir.fr.prj %S2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{kind = "P"}
      %T1 = quiccir.jw.prj %T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{kind = "P"}
      %T1T = quiccir.transpose %T1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
      %T2 = quiccir.al.prj %T1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{kind = "P"}
      %T2T = quiccir.transpose %T2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>>
      %T3 = quiccir.fr.prj %T2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{kind = "D1"}
      return %S3, %T3 : tensor<?x?x?xf64>, tensor<?x?x?xf64>
  }

}
