// RUN: quiccir-opt %s --quiccir-view-wrapper='dim-rets=[1,1,1],[3,3,3] dim-args=[2,2,2] lay-args=argLay0,argLay1 lay-rets=retLay' | FileCheck %s

module {
// CHECK: func.func @_view_entry(%{{.*}}: !llvm.ptr<array<90 x ptr>>, %[[RET0:.*]]: !quiccir.view<1x1x1xf64, "retLay">, %[[RET1:.*]]: !quiccir.view<3x3x3xf64, "retLay">, %[[ARG0:.*]]: !quiccir.view<2x2x2xf64, "argLay0">, %[[ARG1:.*]]: !quiccir.view<2x2x2xf64, "argLay1">) {
// CHECK: %[[TARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !quiccir.view<2x2x2xf64, "argLay0"> to tensor<?x?x?xf64>
// CHECK: %[[TARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !quiccir.view<2x2x2xf64, "argLay1"> to tensor<?x?x?xf64>
// CHECK: %[[RETS:.*]]:2 = call @entry(%[[TARG0]], %[[TARG1]]) : (tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>)
// CHECK: quiccir.materialize %[[RETS]]#0 in %[[RET0]] : (tensor<?x?x?xf64>, !quiccir.view<1x1x1xf64, "retLay">)
func.func private @entry(%arg0: tensor<?x?x?xf64>, %arg1: tensor<?x?x?xf64>) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>)
}
