// RUN: quiccir-opt %s --finalize-quiccir-view | FileCheck %s

// func
module {
  // CHECK: func.func @entry(%{{.*}}: !llvm.ptr<struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>>)
  func.func @entry(%umod: !quiccir.view<1x1x1xf64, "layout">) {
    return
  }
}

// return
module {
  func.func @entry(%umod: !quiccir.view<1x1x1xf64, "layout">) -> (!quiccir.view<1x1x1xf64, "layout">) {
  // CHECK: return %{{.*}}: !llvm.ptr<struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>>
    return %umod: !quiccir.view<1x1x1xf64, "layout">
  }
}

// call
module {
  func.func private @other( !quiccir.view<1x1x1xf64, "layout">)
  func.func @entry(%umod: !quiccir.view<1x1x1xf64, "layout">) {
  // CHECK: call @other(%{{.*}}) : (!llvm.ptr<struct<(array<3 x i32>, ptr<i32>, i32, ptr<i32>, i32, ptr<f64>, i32)>>) -> ()
    call @other(%umod) : (!quiccir.view<1x1x1xf64, "layout">) -> ()
    return
  }
}
