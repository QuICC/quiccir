// RUN: quiccir-opt %s -split-input-file -verify-diagnostics

func.func @use(%ptr: memref<?xi32>, %idx: memref<?xi32>, %data: memref<?xf32>) {
    %view = quiccir.assemble(%ptr, %idx), %data : (memref<?xi32>, memref<?xi32>), memref<?xf32> -> !quiccir.view<1xf32, "layout">
    quiccir.dealloc(%view) : !quiccir.view<1xf32, "layout">
    return %view : !quiccir.view<1xf32, "layout"> // expected-error {{found uses of dealloc operand}}
}

// -----

func.func @funArg(%arg1: !quiccir.view<16x2x3xf32, "layoutUmod">) {
quiccir.dealloc(%arg1) : !quiccir.view<16x2x3xf32, "layoutUmod">  // expected-error {{dealloc operand is a function argument}}
return
}
