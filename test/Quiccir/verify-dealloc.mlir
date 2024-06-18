// RUN: quiccir-opt %s -split-input-file -verify-diagnostics

func.func @use(%arg: !quiccir.view<1xf32, "layout">) -> !quiccir.view<1xf32, "layout"> {
    %view = quiccir.alloc(%arg) : !quiccir.view<1xf32, "layout"> -> !quiccir.view<1xf32, "layout"> {producer = "unknown"}
    quiccir.dealloc(%view) : !quiccir.view<1xf32, "layout">
    return %view : !quiccir.view<1xf32, "layout"> // expected-error {{found uses of dealloc operand}}
}

// -----

func.func @funArg(%arg1: !quiccir.view<16x2x3xf32, "layoutUmod">) {
quiccir.dealloc(%arg1) : !quiccir.view<16x2x3xf32, "layoutUmod">  // expected-error {{dealloc operand is a function argument}}
return
}
