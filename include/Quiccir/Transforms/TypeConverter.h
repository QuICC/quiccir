//===- TypeConverter.hpp - Quiccir type converter ---------------*- C++ -*-===//
//
// This header file defines prototypes of Quiccir type converter.
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_TRANSFORMS_TYPECONVERTER_H
#define QUICCIR_TRANSFORMS_TYPECONVERTER_H

#include "mlir/Transforms/DialectConversion.h"
#include "Quiccir/IR/QuiccirTypes.h"

namespace mlir {
namespace quiccir {

//===----------------------------------------------------------------------===//
// Quiccir View type conversion into llvm struct.
//===----------------------------------------------------------------------===//

/// Quiccir View to a llvm.struct converter.
class ViewTypeToStructConverter : public TypeConverter {
public:
  ViewTypeToStructConverter();

  // !quiccir.view --> llvm.struct<...>
  mlir::Type convertView(ViewType view);
};

/// Quiccir View to a llvm.ptr<struct> converter.
class ViewTypeToPtrOfStructConverter : public TypeConverter {
public:
  ViewTypeToPtrOfStructConverter();

  // !quiccir.view --> llvm.ptr<struct<...>>
  mlir::Type convertView(ViewType view);
};

/// Tensor to Quiccir View converter.
class TensorToViewConverter : public TypeConverter {
public:
  TensorToViewConverter();

  // tensor --> !quiccir.view
  mlir::Type convertTensor(mlir::RankedTensorType tensor);
};

} // namespace quiccir
} // namespace mlir

#endif // QUICCIR_TRANSFORMS_TYPECONVERTER_H