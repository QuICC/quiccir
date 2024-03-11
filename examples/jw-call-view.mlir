// mod -> phys
module {
  func.func private @quiccir_jw_prj(!llvm.ptr, !quiccir.view<1x1x1xf64, "bla">, !quiccir.view<1x1x1xf64, "bla">) attributes {llvm.emit_c_interface}
  func.func @entry(%thisArr: !llvm.array<1x!llvm.ptr>, %uval: !quiccir.view<1x1x1xf64, "bla">, %umod: !quiccir.view<1x1x1xf64, "bla">) attributes {llvm.emit_c_interface} {
    %this = llvm.extractvalue %thisArr [0] : !llvm.array<1x!llvm.ptr>
    call @quiccir_jw_prj(%this, %uval, %umod) : (!llvm.ptr, !quiccir.view<1x1x1xf64, "bla">, !quiccir.view<1x1x1xf64, "bla">) -> ()
    return
  }
}

// ./bin/quiccir-opt ../examples/jw-call.mlir --finalize-quiccir-view --convert-func-to-llvm --canonicalize | ./bin/quiccir-miniapp -emit=jit --shared-libs=./external/libquiccir_external.so
