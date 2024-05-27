// RUN: quiccir-opt %s --set-quiccir-dims='phys=10,10,10 mods=2,2,2' | FileCheck %s

func.func @entry(%arg0: !llvm.ptr<array<5 x ptr>> {llvm.noalias}, %arg1: !quiccir.view<2x2x2xf64, "layMMM">, %arg2: !quiccir.view<10x10x10xf64, "layPPP">) {
  %0 = builtin.unrealized_conversion_cast %arg2 : !quiccir.view<10x10x10xf64, "layPPP"> to tensor<?x?x?xf64>
  // CHECK: %{{.*}}= quiccir.fr.int %{{.*}} : tensor<10x10x10xf64> -> tensor<10x2x10xf64>
  %1 = quiccir.fr.int %0 : tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes {implptr = 0 : i64}
  %2 = quiccir.transpose %1 permutation = [2, 0, 1] : tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes {implptr = 1 : i64}
  // CHECK: %{{.*}}= quiccir.al.int %{{.*}} : tensor<2x10x10xf64> -> tensor<2x2x10xf64>
  %3 = quiccir.al.int %2 : tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes {implptr = 2 : i64}
  %4 = quiccir.transpose %3 permutation = [2, 0, 1] : tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes {implptr = 3 : i64}
  // CHECK: %{{.*}} = quiccir.jw.int %{{.*}} : tensor<2x10x2xf64> -> tensor<2x2x2xf64>
  %5 = quiccir.jw.int %4 : tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes {implptr = 4 : i64}
  quiccir.materialize %5 in %arg1 : (tensor<?x?x?xf64>, !quiccir.view<2x2x2xf64, "layMMM">)
  return
}




