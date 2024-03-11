//===- QuiccirExtension.cpp - Extension module ----------------------------===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#include "Quiccir-c/Dialects.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_quiccirDialects, m) {
  //===--------------------------------------------------------------------===//
  // quiccir dialect
  //===--------------------------------------------------------------------===//
  auto quiccir_m = m.def_submodule("quiccir");

  quiccir_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__quiccir__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
