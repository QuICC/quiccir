func.func @entry(%tumod: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xcomplex<f64>>) {
  %tuval = quiccir.jw.prj %tumod : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 0 :i64}
  %ret = quiccir.jw.int %tuval : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
  return %ret : tensor<?x?x?xcomplex<f64>>
}

// ./bin/quiccir-opt ../examples/simple-1d-loop.mlir --inline --quiccir-view-wrapper='dim-rets=7,3,6 dim-args=7,3,6 lay-args=DCCSC3D_t lay-rets=DCCSC3D_t' --inline --set-quiccir-dims='phys=6,10,10 mods=3,6,7' --set-quiccir-view-lay='lay-ppp2mpp=DCCSC3D_t,DCCSC3D_t lay-pmp2mmp=DCCSC3D_t,S1CLCSC3D_t lay-pmm2mmm=DCCSC3D_t,DCCSC3D_t' --convert-quiccir-to-call --quiccir-view-deallocation --lower-quiccir-alloc --canonicalize --finalize-quiccir-view --convert-func-to-llvm --canonicalize

// ./bin/quiccir-opt ../examples/simple-1d-loop.mlir --inline --quiccir-view-wrapper='dim-rets=7,3,6 dim-args=7,3,6 lay-args=DCCSC3D_t lay-rets=DCCSC3D_t' --inline --set-quiccir-dims='phys=6,10,10 mods=3,6,7' --set-quiccir-view-lay='lay-ppp2mpp=DCCSC3D_t,DCCSC3D_t lay-pmp2mmp=DCCSC3D_t,S1CLCSC3D_t lay-pmm2mmm=DCCSC3D_t,DCCSC3D_t' --convert-quiccir-to-call --convert-quiccir-to-llvm --quiccir-view-deallocation --lower-quiccir-alloc --convert-func-to-llvm --finalize-quiccir-view -convert-func-to-llvm --canonicalize --cse