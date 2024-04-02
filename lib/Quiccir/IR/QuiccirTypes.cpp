//===- QuiccirTypes.cpp - Quiccir dialect types -----------------*- C++ -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#include "Quiccir/IR/QuiccirTypes.h"

#include "Quiccir/IR/QuiccirDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::quiccir;

#define GET_TYPEDEF_CLASSES
#include "Quiccir/IR/QuiccirOpsTypes.cpp.inc"

void QuiccirDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Quiccir/IR/QuiccirOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ViewType
//===----------------------------------------------------------------------===//

ViewType ViewType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                         Type elementType) const {
  return ViewType::get(getContext(), getShape(), elementType, getEncoding());
}


mlir::Type ViewType::parse(AsmParser &parser) {
  // Parse '<'.
  if (parser.parseLess())
    return nullptr;
  // Parse the size and elementType.
  SmallVector<int64_t> shape;
  Type elementType;
  if (parser.parseDimensionList(shape, /*allowDynamic=*/true) ||
      parser.parseType(elementType))
    return nullptr;
  // Parse ','
  if (parser.parseComma())
    return nullptr;

  // Parse encoding.
  Attribute encoding;
  if (parser.parseAttribute(encoding))
    return nullptr;

  // Parse '>'.
  if (parser.parseGreater())
    return nullptr;

  return ViewType::get(parser.getContext(), shape, elementType, encoding);
}

void ViewType::print(AsmPrinter &printer) const {
  printer << "<";
  for (int64_t dim : getShape()) {
    if (dim == ShapedType::kDynamic) {
      printer << '?';
    }
    else {
      printer << dim;
    }
    printer << 'x';
  }
  printer << getElementType();
  printer << ',';
  printer << ' ';
  printer.printStrippedAttrOrType(getEncoding());
  printer << '>';
}

// ::mlir::Type ViewType::parse(::mlir::AsmParser &odsParser) {
//   ::mlir::Builder odsBuilder(odsParser.getContext());
//   ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
//   (void) odsLoc;
//   ::mlir::FailureOr<::llvm::SmallVector<int64_t>> _result_shape;
//   ::mlir::FailureOr<mlir::Type> _result_elementType;
//   ::mlir::FailureOr<Attribute> _result_encoding;
//   // Parse literal '<'
//   if (odsParser.parseLess()) return {};

//   // Parse variable 'shape'
//   _result_shape = ::mlir::FieldParser<::llvm::SmallVector<int64_t>>::parse(odsParser);
//   if (::mlir::failed(_result_shape)) {
//     odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Quiccir_ViewType parameter 'shape' which is to be a `::llvm::ArrayRef<int64_t>`");
//     return {};
//   }

//   // Parse variable 'elementType'
//   _result_elementType = ::mlir::FieldParser<mlir::Type>::parse(odsParser);
//   if (::mlir::failed(_result_elementType)) {
//     odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Quiccir_ViewType parameter 'elementType' which is to be a `mlir::Type`");
//     return {};
