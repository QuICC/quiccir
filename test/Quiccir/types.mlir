// RUN: quiccir-opt %s | FileCheck %s

module {
    func.func @noLds(%a: !quiccir.view<1x2x2xf64, "layout">) -> () {
// CHECK: func.func @noLds(%{{.*}} !quiccir.view<1x2x2xf64, "layout">) {{.*}}
        return
    }

    func.func @lds(%a: !quiccir.view<1x2x2xf64, "layout", lds=1>) -> () {
// CHECK: func.func @lds(%{{.*}} !quiccir.view<1x2x2xf64, "layout", lds = 1 >) {{.*}}
        return
    }
}