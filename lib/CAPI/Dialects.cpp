//===- Dialects.cpp - CAPI for dialects -----------------------------------===//

#include "Quiccir-c/Dialects.h"

#include "Quiccir/IR/QuiccirDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Quiccir, quiccir,
                                      mlir::quiccir::QuiccirDialect)
