// RUN: quiccir-opt %s --convert-quiccir-to-call -split-input-file -verify-diagnostics

module {
    // create new buffer
    func.func @entryTransposeAlloc(%metaArr: !llvm.ptr<array<6 x ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>>>, %thisArr: !llvm.ptr<array<1 x ptr>>, %v: !quiccir.view<16x2x3xf32, "layoutIn">) { // expected-error {{there is no user, cannot identify transform stage}}
    %vt = builtin.unrealized_conversion_cast %v : !quiccir.view<16x2x3xf32, "layoutIn"> to tensor<16x2x3xf32, "layoutIn">
    // expected-error@below {{could not retrieve meta data}}
    // expected-error@below {{failed to legalize operation 'quiccir.transpose' that was explicitly marked illegal}}
    %tra = quiccir.transpose %vt permutation = [0, 2, 1] : tensor<16x2x3xf32, "layoutIn"> -> tensor<16x3x2xf32, "layoutOut"> attributes{implptr = 0 :i64}
    return
    }
}
