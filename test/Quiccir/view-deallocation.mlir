// RUN: quiccir-opt -pass-pipeline='builtin.module(builtin.module(func.func(quiccir-view-deallocation)))' %s -verify-diagnostics
// RUN: quiccir-opt -pass-pipeline='builtin.module(builtin.module(func.func(quiccir-view-deallocation)))' %s | FileCheck %s

module {
    func.func @unused(%ptr: memref<?xi32>, %idx: memref<?xi32>) {
        %lds = llvm.mlir.constant(3 : i64) : i64
        %data = quiccir.alloc_data(%ptr, %idx), %lds : (memref<?xi32>, memref<?xi32>), i64 -> memref<?xf32> {layout = "layout"}
        %view = quiccir.assemble(%ptr, %idx), %data : (memref<?xi32>, memref<?xi32>), memref<?xf32> -> !quiccir.view<1xf32, "layout"> // expected-warning {{trying to deallocate unused view}}
        return
    }
}

module {
    func.func private @user(!quiccir.view<1xf32, "layout">)
    func.func @allocData(%ptr: memref<?xi32>, %idx: memref<?xi32>) {
        %lds = llvm.mlir.constant(3 : i64) : i64
        %data = quiccir.alloc_data(%ptr, %idx), %lds : (memref<?xi32>, memref<?xi32>), i64 -> memref<?xf32> {layout = "layout"}
        %view = quiccir.assemble(%ptr, %idx), %data : (memref<?xi32>, memref<?xi32>), memref<?xf32> -> !quiccir.view<1xf32, "layout">
        call @user(%view) : (!quiccir.view<1xf32, "layout">) -> ()
        // CHECK: call @user(%[[V:.*]]) : (!quiccir.view<1xf32, "layout">) -> ()
        // CHECK: quiccir.dealloc(%[[V]]) : !quiccir.view<1xf32, "layout">
        return
    }
}

module {
    func.func private @user(!quiccir.view<1xf32, "layout">)
    func.func @allocData(%ptr: memref<?xi32>, %idx: memref<?xi32>, %data: memref<?xf32>) {
        %view = quiccir.assemble(%ptr, %idx), %data : (memref<?xi32>, memref<?xi32>), memref<?xf32> -> !quiccir.view<1xf32, "layout">
        call @user(%view) : (!quiccir.view<1xf32, "layout">) -> ()
        // CHECK: call @user(%[[V:.*]]) : (!quiccir.view<1xf32, "layout">) -> ()
        // CHECK-NEXT: return
        return
    }
}