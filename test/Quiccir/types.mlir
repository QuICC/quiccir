// RUN: quiccir-opt %s | FileCheck %s

module {
    func.func @entry(%a: !quiccir.view<1x2x2xf64, "layout">) -> () {
// CHECK: func.func @entry(%{{.*}} !quiccir.view<1x2x2xf64, "layout">) {{.*}}
        return
    }
}