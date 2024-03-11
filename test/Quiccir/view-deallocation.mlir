// RUN: quiccir-opt -pass-pipeline='builtin.module(builtin.module(func.func(quiccir-view-deallocation)))' %s -verify-diagnostics
// RUN: quiccir-opt -pass-pipeline='builtin.module(builtin.module(func.func(quiccir-view-deallocation)))' %s | FileCheck %s
module {
    func.func @wrap(%arg: !quiccir.view<1xf32, "layout">) {
        %view = quiccir.alloc(%arg) : !quiccir.view<1xf32, "layout"> -> !quiccir.view<1xf32, "layout"> {producer = "unknown"} // expected-warning {{trying to deallocate unused view}}
        return
    }
}

module {
    func.func private @user(!quiccir.view<1xf32, "layout">)
    func.func @wrap(%arg: !quiccir.view<1xf32, "layout">) {
        %view = quiccir.alloc(%arg) : !quiccir.view<1xf32, "layout"> -> !quiccir.view<1xf32, "layout"> {producer = "unknown"}
        call @user(%view) : (!quiccir.view<1xf32, "layout">) -> ()
        // CHECK: quiccir.dealloc(%{{.*}}) : !quiccir.view<1xf32, "layout">
        return
    }
}