// RUN: quiccir-opt %s -verify-diagnostics
module {
    func.func @wrap(%arg: !quiccir.view<1xf32, "layout">) -> !quiccir.view<1xf32, "layout"> {
        %view = quiccir.alloc(%arg) : !quiccir.view<1xf32, "layout"> -> !quiccir.view<1xf32, "layout"> {producer = "unknown"}
        quiccir.dealloc(%view) : !quiccir.view<1xf32, "layout">
        return %view : !quiccir.view<1xf32, "layout"> // expected-error {{found uses of dealloc operand}}
    }
}