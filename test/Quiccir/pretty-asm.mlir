// RUN: quiccir-opt %s | FileCheck %s

module {
    func.func @wrap(%arg: !quiccir.view<1xf32, "layout">) {
        // CHECK: %{{.*}} = quiccir.alloc(%{{.*}}) : !quiccir.view<1xf32, "layout"> -> !quiccir.view<1xf32, "layout"> {producer = "unknown"}
        %view = quiccir.alloc(%arg) : !quiccir.view<1xf32, "layout"> -> !quiccir.view<1xf32, "layout"> {producer = "unknown"}
        return
    }
}

module {
    func.func @wrap(%arg: !quiccir.view<1xf32, "layout">) {
        %view = quiccir.alloc(%arg) : !quiccir.view<1xf32, "layout"> -> !quiccir.view<1xf32, "layout"> {producer = "unknown"}
        // CHECK: quiccir.dealloc(%{{.*}}) : !quiccir.view<1xf32, "layout">
        quiccir.dealloc(%view) : !quiccir.view<1xf32, "layout">
        return
    }
}

module {
    func.func @wrap(%arg: !quiccir.view<1xf32, "layout">)  {
        %c0 = arith.constant 1.0 : f32
        %t = tensor.splat %c0 : tensor<1xf32, "layout">
        %v = quiccir.alloc(%arg) : !quiccir.view<1xf32, "layout"> -> !quiccir.view<1xf32, "layout"> {producer = "unknown"}
        // CHECK: quiccir.materialize %{{.*}} in %{{.*}} : (tensor<1xf32, "layout">, !quiccir.view<1xf32, "layout">)
        quiccir.materialize %t in %v : (tensor<1xf32, "layout">, !quiccir.view<1xf32, "layout">)
        return
    }
}

module {
    func.func @simple_no_arg() {
        %ptr = memref.alloc() : memref<4xi32>
        %idx = memref.alloc() : memref<4xi32>
        // CHECK: %{{.*}} = quiccir.alloc_data(%{{.*}}, %{{.*}}) : (memref<4xi32>, memref<4xi32>) -> memref<4xf64> {layout = "unknown"}
        %view = quiccir.alloc_data(%ptr, %idx) : (memref<4xi32>, memref<4xi32>) -> memref<4xf64> {layout = "unknown"}
        return
    }
}

module {
    func.func @simple_no_arg() {
        %ptr = memref.alloc() : memref<4xi32>
        %idx = memref.alloc() : memref<4xi32>
        %data = memref.alloc() : memref<4xf32>
        // CHECK: %{{.*}} = quiccir.assemble(%{{.*}}, %{{.*}}), %{{.*}} : (memref<4xi32>, memref<4xi32>), memref<4xf32> -> !quiccir.view<?x?x?xf32, "unknown">
        %view = quiccir.assemble(%ptr, %idx), %data : (memref<4xi32>, memref<4xi32>), memref<4xf32> -> !quiccir.view<?x?x?xf32, "unknown">
        return
    }
}

module {
    func.func @wrap(%arg: !quiccir.view<1xf32, "layout">)  {
        // CHECK: quiccir.pointers %{{.*}} : !quiccir.view<1xf32, "layout"> -> memref<?xi32>
        %ptr = quiccir.pointers %arg : !quiccir.view<1xf32, "layout"> -> memref<?xi32>
        return
    }
}

module {
    func.func @wrap(%arg: !quiccir.view<1xf32, "layout">)  {
        // CHECK: quiccir.indices %{{.*}} : !quiccir.view<1xf32, "layout"> -> memref<?xi32>
        %idx = quiccir.indices %arg : !quiccir.view<1xf32, "layout"> -> memref<?xi32>
        return
    }
}

module {
    func.func @simple_no_arg() -> tensor<16x3x2xf32> {
        %c0 = arith.constant 1.0 : f32
        %op = tensor.splat %c0 : tensor<16x3x2xf32>
        %umod = tensor.splat %c0 : tensor<16x2x2xf32>
        // CHECK: %{{.*}} = quiccir.quadrature %{{.*}}, %{{.*}} : tensor<16x3x2xf32>, tensor<16x2x2xf32> -> tensor<16x3x2xf32>
        %0 = quiccir.quadrature %op, %umod : tensor<16x3x2xf32>, tensor<16x2x2xf32> -> tensor<16x3x2xf32>
        return %0 : tensor<16x3x2xf32>
    }
}

module {
    func.func @wrap(%arg0: tensor<?x?x?xcomplex<f32>>) -> tensor<?x?x?xf32> {
        // CHECK: %{{.*}} = quiccir.fr.prj %{{.*}} : tensor<?x?x?xcomplex<f32>> -> tensor<?x?x?xf32>
        %0 = quiccir.fr.prj %arg0 : tensor<?x?x?xcomplex<f32>> -> tensor<?x?x?xf32>
        return %0 : tensor<?x?x?xf32>
    }
}

module {
    func.func @wrap(%arg0: tensor<?x?x?xcomplex<f32>>) -> tensor<?x?x?xf32> {
        // CHECK: %{{.*}} = quiccir.fr.prj %{{.*}} : tensor<?x?x?xcomplex<f32>> -> tensor<?x?x?xf32> attributes {implptr = {{.*}} : i64}
        %0 = quiccir.fr.prj %arg0 : tensor<?x?x?xcomplex<f32>> -> tensor<?x?x?xf32> attributes {implptr = 2 : i64}
        return %0 : tensor<?x?x?xf32>
    }
}

module {
    func.func @wrap(%arg0: tensor<?x?x?xcomplex<f32>>) -> tensor<?x?x?xcomplex<f32>> {
        // CHECK: %{{.*}} = quiccir.jw.prj %{{.*}} : tensor<?x?x?xcomplex<f32>> -> tensor<?x?x?xcomplex<f32>>
        %0 = quiccir.jw.prj %arg0 : tensor<?x?x?xcomplex<f32>> -> tensor<?x?x?xcomplex<f32>>
        return %0 : tensor<?x?x?xcomplex<f32>>
    }
}

module {
    func.func @wrap(%arg0: tensor<?x?x?xcomplex<f32>>) -> tensor<?x?x?xcomplex<f32>> {
        // CHECK: %{{.*}} = quiccir.jw.prj %{{.*}} : tensor<?x?x?xcomplex<f32>> -> tensor<?x?x?xcomplex<f32>> attributes {implptr = {{.*}} : i64}
        %0 = quiccir.jw.prj %arg0 : tensor<?x?x?xcomplex<f32>> -> tensor<?x?x?xcomplex<f32>> attributes {implptr = 2 : i64}
        return %0 : tensor<?x?x?xcomplex<f32>>
    }
}
