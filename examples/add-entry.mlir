// mod -> phys
!type_uval = !quiccir.view<4x10x6xf64, "layoutUval">
!type_umod = !quiccir.view<5x2x3xf64, "layoutUmod">

!type_tuval = tensor<4x10x6xf64, "layoutUval">
!type_tumod = tensor<5x2x3xf64, "layoutUmod">


func.func private @tensor_entry(%Polur: tensor<2x2x2xf64, "lay">) -> (tensor<1x1x1xf64, "lay">)

// func.func @entry(%thisArr: !llvm.ptr<array<15 x ptr>> {llvm.noalias}, %PolNewv: !type_umod, %Polv: !type_umod) {
//   %Pol = builtin.unrealized_conversion_cast %Polv : !type_umod to !type_tumod
//   %Polur = tensor.cast %Pol : !type_tumod to tensor<?x?x?xf64>
//   %PolNewur = call @tensor_entry(%Polur) : (tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
//   %PolNew = tensor.cast %PolNewur : tensor<?x?x?xf64> to !type_tumod
//   /// if this is the only consumer write to existing buffer
//   quiccir.materialize %PolNew in %PolNewv : (!type_tumod, !type_umod)
//   return
// }
// ./bin/quiccir-opt ../examples/add-entry.mlir --inline
// add entry pass
// in/out types
