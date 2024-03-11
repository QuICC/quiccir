//===- Dialects.h - CAPI for dialects -----------------------------*- C -*-===//

#ifndef QUICCIR_C_DIALECTS_H
#define QUICCIR_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Quiccir, quiccir);

#ifdef __cplusplus
}
#endif

#endif // QUICCIR_C_DIALECTS_H
