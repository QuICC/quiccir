# RUN: %python %s | FileCheck %s

from mlir_quiccir.ir import *
from mlir_quiccir.dialects import (
  builtin as builtin_d,
  quiccir as quiccir_d
)

with Context():
  quiccir_d.register_dialect()
  module = Module.parse("""
    %c0 = arith.constant 1.0 : f32
    %b0 = tensor.splat %c0 : tensor<2x2xf32>
    %b1 = tensor.splat %c0 : tensor<2x2xf32>
    %umod = tensor.splat %c0 : tensor<2x2xf32>
    %0 = quiccir.quadrature %b0, %b1, %umod : tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    """)
  # CHECK: %[[C:.*]] = arith.constant 1.000000e+00 : f32
  # CHECK: %{{.*}} = quiccir.quadrature %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
  print(str(module))
