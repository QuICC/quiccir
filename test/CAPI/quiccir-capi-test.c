//===- quiccir-cap-demo.c - Simple demo of C-API -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: quiccir-capi-test 2>&1 | FileCheck %s

#include <stdio.h>

#include "Quiccir-c/Dialects.h"
#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"

static void registerAllUpstreamDialects(MlirContext ctx) {
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterAllDialects(registry);
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirDialectRegistryDestroy(registry);
}

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  // TODO: Create the dialect handles for the builtin dialects and avoid this.
  // This adds dozens of MB of binary size over just the quiccir dialect.
  registerAllUpstreamDialects(ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__quiccir__(), ctx);

  MlirModule module = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString(
        "%c0 = arith.constant 1.0 : f32\n"
        "%op = tensor.splat %c0 : tensor<16x2x2xf32>\n"
        "%umod = tensor.splat %c0 : tensor<16x2x2xf32>\n"
        "%0 = quiccir.quadrature %op, %umod : tensor<16x2x2xf32>, tensor<16x2x2xf32> -> tensor<16x2x2xf32>\n"));
  if (mlirModuleIsNull(module)) {
    printf("ERROR: Could not parse.\n");
    mlirContextDestroy(ctx);
    return 1;
  }
  MlirOperation op = mlirModuleGetOperation(module);

  // CHECK: %[[C:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %{{.*}} = quiccir.quadrature %{{.*}}, %{{.*}} : tensor<16x2x2xf32>, tensor<16x2x2xf32> -> tensor<16x2x2xf32>
  mlirOperationDump(op);

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
  return 0;
}
