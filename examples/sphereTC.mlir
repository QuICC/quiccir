func.func private @bwdScalar(%S: tensor<?x?x?xcomplex<f64>>) -> tensor<?x?x?xf64> {
    // backward scalar path
    %S1 = quiccir.jw.prj %S : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 0 :i64, kind = "P"}
    %S1T = quiccir.transpose %S1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
    %S2 = quiccir.al.prj %S1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 2 :i64, kind = "P"}
    %S2T = quiccir.transpose %S2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 3 :i64}
    %S3 = quiccir.fr.prj %S2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{implptr = 4 :i64, kind = "P"}
    return %S3 : tensor<?x?x?xf64>
}

func.func private @bwdGradScalar(%S: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) {
    // grad R
    %SdR1 = quiccir.jw.prj %S : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 10 :i64, kind = "D1"}
    %SdR1T = quiccir.transpose %SdR1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
    %SdR2 = quiccir.al.prj %SdR1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 2 :i64, kind = "P"}
    %SdR2T = quiccir.transpose %SdR2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 3 :i64}
    %SdR = quiccir.fr.prj %SdR2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{implptr = 4 :i64, kind = "P"}
    // grad Theta
    %SdTh1 = quiccir.jw.prj %S : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 11 :i64, kind = "DivR1_Zero"}
    %SdTh1T = quiccir.transpose %SdTh1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
    %SdTh2 = quiccir.al.prj %SdTh1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 12 :i64, kind = "D1"}
    %SdTh2T = quiccir.transpose %SdTh2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 3 :i64}
    %SdTh = quiccir.fr.prj %SdTh2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{implptr = 4 :i64, kind = "P"}
    // grad Phi
    %SdPh1 = quiccir.jw.prj %S : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 11 :i64, kind = "DivR1_Zero"}
    %SdPh1T = quiccir.transpose %SdPh1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
    %SdPh2 = quiccir.al.prj %SdPh1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 13 :i64, kind = "DivS1Dp"}
    %SdPh2T = quiccir.transpose %SdPh2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 3 :i64}
    %SdPh = quiccir.fr.prj %SdPh2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{implptr = 4 :i64, kind = "P"}
    return %SdR, %SdTh, %SdPh : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>
}

func.func private @bwdVector(%Tor: tensor<?x?x?xcomplex<f64>>, %Pol: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) {
    // R
    %PolR1 = quiccir.jw.prj %Pol : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 11 :i64, kind = "DivR1_Zero"}
    %PolR1T = quiccir.transpose %PolR1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
    %PolR2 = quiccir.al.prj %PolR1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 20 :i64, kind = "Ll"}
    %PolR2T = quiccir.transpose %PolR2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 3 :i64}
    %R = quiccir.fr.prj %PolR2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{implptr = 4 :i64, kind = "P"}
    // Theta
    %TorTh1 = quiccir.jw.prj %Tor : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 0 :i64, kind = "P"}
    %TorTh1T = quiccir.transpose %TorTh1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
    %TorTh2 = quiccir.al.prj %TorTh1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 13 :i64, kind = "DivS1Dp"}
    %TorTh2T = quiccir.transpose %TorTh2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 3 :i64}
    %TorTh3 = quiccir.fr.prj %TorTh2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{implptr = 4 :i64, kind = "P"}
    //
    %PolTh1 = quiccir.jw.prj %Pol : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 11 :i64, kind = "DivR1D1R1_Zero"}
    %PolTh1T = quiccir.transpose %PolTh1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
    %PolTh2 = quiccir.al.prj %PolTh1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 21 :i64, kind = "D1"}
    %PolTh2T = quiccir.transpose %PolTh2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 3 :i64}
    %PolTh3 = quiccir.fr.prj %PolTh2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{implptr = 4 :i64, kind = "P"}
    //
    %Theta = quiccir.add %TorTh3, %PolTh3 : tensor<?x?x?xf64>, tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes{implptr = 22 :i64}
    // Phi
    %TorPh1 = quiccir.jw.prj %Tor : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 0 :i64, kind = "P"}
    %TorPh1T = quiccir.transpose %TorPh1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
    %TorPh2 = quiccir.al.prj %TorPh1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 21 :i64, kind = "D1"}
    %TorPh2T = quiccir.transpose %TorPh2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 3 :i64}
    %TorPh3 = quiccir.fr.prj %TorPh2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{implptr = 4 :i64, kind = "P"}
    //
    %PolPh1 = quiccir.jw.prj %Pol : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 11 :i64, kind = "DivR1D1R1_Zero"}
    %PolPh1T = quiccir.transpose %PolPh1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
    %PolPh2 = quiccir.al.prj %PolPh1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 13 :i64, kind = "DivS1Dp"}
    %PolPh2T = quiccir.transpose %PolPh2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 3 :i64}
    %PolPh3 = quiccir.fr.prj %PolPh2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{implptr = 4 :i64, kind = "P"}
    //
    %Phi = quiccir.sub %PolPh3, %TorPh3 : tensor<?x?x?xf64>, tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes{implptr = 22 :i64}
    return %R, %Theta, %Phi : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>
}

func.func private @bwdCurl(%Tor: tensor<?x?x?xcomplex<f64>>, %Pol: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) {
    // R
    %TorR1 = quiccir.jw.prj %Tor : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 11 :i64, kind = "DivR1_Zero"}
    %TorR1T = quiccir.transpose %TorR1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
    %TorR2 = quiccir.al.prj %TorR1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 20 :i64, kind = "Ll"}
    %TorR2T = quiccir.transpose %TorR2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 3 :i64}
    %R = quiccir.fr.prj %TorR2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{implptr = 4 :i64, kind = "P"}
    // Theta
    %TorTh1 = quiccir.jw.prj %Tor : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 11 :i64, kind = "DivR1D1R1_Zero"}
    %TorTh1T = quiccir.transpose %TorTh1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
    %TorTh2 = quiccir.al.prj %TorTh1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 21 :i64, kind = "D1"}
    %TorTh2T = quiccir.transpose %TorTh2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 3 :i64}
    %TorTh3 = quiccir.fr.prj %TorTh2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{implptr = 4 :i64, kind = "P"}
    //
    %PolTh1 = quiccir.jw.prj %Pol : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 30 :i64, kind = "SphLapl"}
    %PolTh1T = quiccir.transpose %PolTh1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
    %PolTh2 = quiccir.al.prj %PolTh1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 13 :i64, kind = "DivS1Dp"}
    %PolTh2T = quiccir.transpose %PolTh2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 3 :i64}
    %PolTh3 = quiccir.fr.prj %PolTh2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{implptr = 4 :i64, kind = "P"}
    //
    %Theta = quiccir.sub %TorTh3, %PolTh3 : tensor<?x?x?xf64>, tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes{implptr = 22 :i64}
    // Phi
    %TorPh1 = quiccir.jw.prj %Tor : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 11 :i64, kind = "DivR1D1R1_Zero"}
    %TorPh1T = quiccir.transpose %TorPh1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
    %TorPh2 = quiccir.al.prj %TorPh1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 13 :i64, kind = "DivS1Dp"}
    %TorPh2T = quiccir.transpose %TorPh2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 3 :i64}
    %TorPh3 = quiccir.fr.prj %TorPh2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{implptr = 4 :i64, kind = "P"}
    //
    %PolPh1 = quiccir.jw.prj %Pol : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 30 :i64, kind = "SphLapl"}
    %PolPh1T = quiccir.transpose %PolPh1 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 1 :i64}
    %PolPh2 = quiccir.al.prj %PolPh1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 21 :i64, kind = "D1"}
    %PolPh2T = quiccir.transpose %PolPh2 permutation = [1, 2, 0] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 3 :i64}
    %PolPh3 = quiccir.fr.prj %PolPh2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xf64> attributes{implptr = 4 :i64, kind = "P"}
    //
    %Phi = quiccir.add %PolPh3, %TorPh3 : tensor<?x?x?xf64>, tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes{implptr = 22 :i64}
    return %R, %Theta, %Phi : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>
}

func.func private @fwdScalar(%S: tensor<?x?x?xf64>) -> tensor<?x?x?xcomplex<f64>> {
    // forward scalar Nl path
    %S1 = quiccir.fr.int %S : tensor<?x?x?xf64> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 40 :i64, kind = "P"}
    %S1T = quiccir.transpose %S1 permutation = [2, 0, 1] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 41 :i64}
    %S2 = quiccir.al.int %S1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 42 :i64, kind = "P"}
    %S2T = quiccir.transpose %S2 permutation = [2, 0, 1] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 43 :i64}
    %S3 = quiccir.jw.int %S2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 44 :i64, kind = "I2"}

    return %S3 : tensor<?x?x?xcomplex<f64>>
}

func.func private @fwdVector(%R: tensor<?x?x?xf64>, %Theta: tensor<?x?x?xf64>, %Phi: tensor<?x?x?xf64>) -> (tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>){
    // Tor
    %ThetaTor1 = quiccir.fr.int %Theta : tensor<?x?x?xf64> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 40 :i64, kind = "P"}
    %ThetaTor1T = quiccir.transpose %ThetaTor1 permutation = [2, 0, 1] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 41 :i64}
    %ThetaTor2 = quiccir.al.int %ThetaTor1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 50 :i64, kind = "DivLlDivS1Dp"}
    %ThetaTor2T = quiccir.transpose %ThetaTor2 permutation = [2, 0, 1] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 43 :i64}
    %ThetaTor3 = quiccir.jw.int %ThetaTor2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 44 :i64, kind = "P"}
    //
    %PhiTor1 = quiccir.fr.int %Phi : tensor<?x?x?xf64> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 40 :i64, kind = "P"}
    %PhiTor1T = quiccir.transpose %PhiTor1 permutation = [2, 0, 1] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 41 :i64}
    %PhiTor2 = quiccir.al.int %PhiTor1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 51 :i64, kind = "DivLlD1"}
    %PhiTor2T = quiccir.transpose %PhiTor2 permutation = [2, 0, 1] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 43 :i64}
    %PhiTor3 = quiccir.jw.int %PhiTor2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 44 :i64, kind = "P"}
    //
    %Tor = quiccir.sub %ThetaTor3, %PhiTor3 : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 52 :i64}

    // Pol
    %RPol1 = quiccir.fr.int %R : tensor<?x?x?xf64> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 40 :i64, kind = "P"}
    %RPol1T = quiccir.transpose %RPol1 permutation = [2, 0, 1] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 41 :i64}
    %RPol2 = quiccir.al.int %RPol1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 42 :i64, kind = "P"}
    %RPol2T = quiccir.transpose %RPol2 permutation = [2, 0, 1] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 43 :i64}
    %RPol3 = quiccir.jw.int %RPol2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 53 :i64, kind = "I4DivR1_Zero"}
    //
    %ThetaPol1 = quiccir.fr.int %Theta : tensor<?x?x?xf64> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 40 :i64, kind = "P"}
    %ThetaPol1T = quiccir.transpose %ThetaPol1 permutation = [2, 0, 1] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 41 :i64}
    %ThetaPol2 = quiccir.al.int %ThetaPol1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 51 :i64, kind = "DivLlD1"}
    %ThetaPol2T = quiccir.transpose %ThetaPol2 permutation = [2, 0, 1] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 43 :i64}
    %ThetaPol3 = quiccir.jw.int %ThetaPol2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 54 :i64, kind = "I4DivR1D1R1_Zero"}
    //
    %PhiPol1 = quiccir.fr.int %Phi : tensor<?x?x?xf64> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 40 :i64, kind = "P"}
    %PhiPol1T = quiccir.transpose %PhiPol1 permutation = [2, 0, 1] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 41 :i64}
    %PhiPol2 = quiccir.al.int %PhiPol1T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 50 :i64, kind = "DivLlDivS1Dp"}
    %PhiPol2T = quiccir.transpose %PhiPol2 permutation = [2, 0, 1] : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 43 :i64}
    %PhiPol3 = quiccir.jw.int %PhiPol2T : tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 54 :i64, kind = "I4DivR1D1R1_Zero"}
    //
    %tmp = quiccir.add %ThetaPol3, %PhiPol3 : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 53 :i64}
    %Pol = quiccir.sub %tmp, %RPol3 : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>> -> tensor<?x?x?xcomplex<f64>> attributes{implptr = 52 :i64}
    return %Tor, %Pol : tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
}

func.func private @nlScalar(%UR: tensor<?x?x?xf64>, %UTheta: tensor<?x?x?xf64>, %UPhi: tensor<?x?x?xf64>,
    %TdR: tensor<?x?x?xf64>, %TdTheta: tensor<?x?x?xf64>, %TdPhi: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    // // U dot grad T
    // %DotT = quiccir.dot(%UR, %UTheta, %UPhi, %TdTR, %TdTTheta, %TdTPhi) :
    //     (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) ->
    //     (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>)
    //     attributes{implptr = 60}
    // // U dot R
    // %DotR = quiccir.mul.const(%UR) : (tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
    //     attributes{implptr = 61, kind = "R"}
    // %TPhysNl = quiccir.sub %Dot, %DotR : tensor<?x?x?xf64>, tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes{implptr = 62}

    // return %TPhysNl : tensor<?x?x?xf64>
    return %TdR : tensor<?x?x?xf64>
}

func.func private @nlVector(%UR: tensor<?x?x?xf64>, %UTheta: tensor<?x?x?xf64>, %UPhi: tensor<?x?x?xf64>,
    %CurlR: tensor<?x?x?xf64>, %CurlTheta: tensor<?x?x?xf64>, %CurlPhi: tensor<?x?x?xf64>, %T: tensor<?x?x?xf64>) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) {
    // Cross
    %Cross:3 = quiccir.cross(%UR, %UTheta, %UPhi), (%CurlR, %CurlTheta, %CurlPhi) :
        (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>), (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) ->
        (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>)
        attributes{implptr = 61, kind = "inertia"}
    // // Add buoyancy
    // %Buoy = quiccir.mul.const(%T) : (tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
    //     attributes{implptr = 61, kind = "R"}
    // %RNl = quiccir.sub(%Cross#0, %Buoy) : tensor<?x?x?xf64>, tensor<?x?x?xf64> -> tensor<?x?x?xf64> attributes{implptr = 63}
    // return %Rnl, %Cross#1, %Cross#2 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>
    return %Cross#0, %Cross#1, %Cross#2 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>
}

func.func @entry(%T: tensor<?x?x?xcomplex<f64>>, %Tor: tensor<?x?x?xcomplex<f64>>, %Pol: tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>) {
    %TPhys = call @bwdScalar(%T) : (tensor<?x?x?xcomplex<f64>>) -> tensor<?x?x?xf64>
    %TGrad:3 = call @bwdGradScalar(%T) : (tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>)
    %Vel:3 = call @bwdVector(%Tor, %Pol) : (tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>)
    %Curl:3 = call @bwdCurl(%Tor, %Pol) : (tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>)
    %TPhysNl = call @nlScalar(%Vel#0, %Vel#1, %Vel#2, %TGrad#0, %TGrad#1, %TGrad#2) : (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
    %VelNl:3 = call @nlVector(%Vel#0, %Vel#1, %Vel#2, %Curl#0, %Curl#1, %Curl#2, %TPhys) : (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>)
    %TNl = call @fwdScalar(%TPhysNl) : (tensor<?x?x?xf64>) -> tensor<?x?x?xcomplex<f64>>
    %TorNl, %PolNl = call @fwdVector(%VelNl#0, %VelNl#1, %VelNl#2) : (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> (tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>)
    return %TNl, %TorNl, %PolNl: tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>, tensor<?x?x?xcomplex<f64>>
}


// ./bin/quiccir-opt ../examples/sphereTC.mlir --inline --quiccir-view-wrapper='dim-rets=7,3,6 dim-args=7,3,6 lay-args=C_DCCSC3D_t lay-rets=C_DCCSC3D_t' --inline --set-quiccir-dims='phys=6,10,10 mods=3,6,7' --set-quiccir-view-lay='lay-ppp2mpp=R_DCCSC3D_t,C_DCCSC3D_t lay-pmp2mmp=C_DCCSC3D_t,C_S1CLCSC3D_t lay-pmm2mmm=C_DCCSC3D_t,C_DCCSC3D_t' --canonicalize --convert-quiccir-to-call --quiccir-view-deallocation --convert-quiccir-to-llvm --lower-quiccir-alloc --canonicalize --finalize-quiccir-view --convert-func-to-llvm --canonicalize --cse