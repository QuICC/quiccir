// mod -> phys
!type_umod = !quiccir.view<1x2x3xf64, "layoutUmod">
!type_uval = !quiccir.view<1x3x3xf64, "layoutUval">

!type_tumod = tensor<1x2x3xf64, "layoutUmod">
!type_tuval = tensor<1x3x3xf64, "layoutUval">

func.func @simpleTree(%R: tensor<?x?x?xf64>, %Theta: tensor<?x?x?xf64>, %Phi: tensor<?x?x?xf64>) -> (tensor<?x?x?xf64>) {
  // // R
  // %R1 = quiccir.fr.int %R : tensor<?x?x?xf64> -> tensor<?x?x?xf64>
  // %R2 = quiccir.al.int %R1 : tensor<?x?x?xf64> -> tensor<?x?x?xf64>
  // %R3 = quiccir.jw.int %R2 : tensor<?x?x?xf64> -> tensor<?x?x?xf64>
  // // Theta
  // %Th1 = quiccir.fr.int %Theta : tensor<?x?x?xf64> -> tensor<?x?x?xf64>
  // %Th2 = quiccir.al.int %Th1 : tensor<?x?x?xf64> -> tensor<?x?x?xf64>
  // %Th3 = quiccir.jw.int %Th2 : tensor<?x?x?xf64> -> tensor<?x?x?xf64>
  // // Phi
  // %Phi1 = quiccir.fr.int %Phi : tensor<?x?x?xf64> -> tensor<?x?x?xf64>
  // %Phi2 = quiccir.al.int %Phi1 : tensor<?x?x?xf64> -> tensor<?x?x?xf64>
  // %Phi3 = quiccir.jw.int %Phi2 : tensor<?x?x?xf64> -> tensor<?x?x?xf64>

  // %0 = tensor.empty() : tensor<*xf64>
  %Phi3 = linalg.add ins(%R, %Theta : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%Phi: tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
  return %Phi3 : tensor<?x?x?xf64>

  // Pol
  // %tmp = "new.sub"(%Th3, %R3) : (tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
  // %Pol = "new.add"(%tmp, %Phi3) : (tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> tensor<?x?x?xf64>

  // return %Pol : tensor<?x?x?xf64>
}

func.func @entry(%thisArr: !llvm.array<1x!llvm.ptr>, %Polv: !type_umod, %Rv: !type_uval, %Thetav: !type_uval, %Phiv: !type_uval) {
  %R = builtin.unrealized_conversion_cast %Rv : !type_uval to !type_tuval
  %Theta = builtin.unrealized_conversion_cast %Thetav : !type_uval to !type_tuval
  %Phi = builtin.unrealized_conversion_cast %Phiv : !type_uval to !type_tuval
  %RR = tensor.cast %R : !type_tuval to tensor<?x?x?xf64>
  %TTheta = tensor.cast %Theta : !type_tuval to tensor<?x?x?xf64>
  %PPhi = tensor.cast %Phi : !type_tuval to tensor<?x?x?xf64>
  %Polt = call @simpleTree(%RR, %TTheta, %PPhi) : (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
  %Pol = tensor.cast %Polt : tensor<?x?x?xf64> to !type_tumod
  /// if this is the only consumer write to existing buffer
  quiccir.materialize %Pol in %Polv : (!type_tumod, !type_umod)
  return
}


// ./bin/quiccir-opt ../examples/simple-tree.mlir --convert-quiccir-to-call --canonicalize --finalize-quiccir-view --convert-func-to-llvm --canonicalize
