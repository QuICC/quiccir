//===-- QuiccirPasses.td - Quiccir pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_QUICCIRPASSES
#define QUICCIR_QUICCIRPASSES

include "mlir/Pass/PassBase.td"

// def QuiccirLowerToAffine : Pass<"convert-quiccir-to-affine", "func::FuncOp"> {
//   let summary = "Lower the operations from the quiccir dialect into affine "
//                 "loops";
//   let constructor = "mlir::quiccir::createLowerToAffinePass()";
//   let dependentDialects = [
//     "AffineDialect",
//     "func::FuncDialect",
//     "memref::MemRefDialect",
//     "bufferization::BufferizationDialect"];
// }

// def QuiccirLowerToLinalg : Pass<"convert-quiccir-to-linalg", "func::FuncOp"> {
//   let summary = "Lower the operations from the quiccir dialect into linalg "
//                 "loops";
//   let constructor = "mlir::quiccir::createLowerToLinalgPass()";
//   let dependentDialects = [
//     "tosa::TosaDialect",
//     "AffineDialect",
//     "linalg::LinalgDialect",
//     "func::FuncDialect",
//     "memref::MemRefDialect",
//     "bufferization::BufferizationDialect"];
// }

#endif // QUICCIR_QUICCIRPASSES
