//===- TypeConverter.cpp - Quiccir type converter ---------------*- C++ -*-===//

#include "Quiccir/Transforms/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

//===----------------------------------------------------------------------===//
// Quiccir View type conversion into llvm struct
//===----------------------------------------------------------------------===//

using namespace mlir::quiccir;

ViewTypeToStructConverter::ViewTypeToStructConverter() {
  addConversion([](Type type) { return type; });
  addConversion([&](ViewType view) -> Type {
    return convertView(view);
  });

  // Add generic source and target materializations to handle cases where
  // non-LLVM types persist after an LLVM conversion.
  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

mlir::Type ViewTypeToStructConverter::convertView(ViewType view) {
    /// Types contained by the struct dims[?],
    /// pos prt, pos size, coo ptr, coo size, data ptr, data size
    /// type of meta data could be an optional attribute
    /// probably need to add address space attribute to view.
    auto *ctx = view.getContext();
    SmallVector<mlir::Type, 7> structElementTypes;
    mlir::Type i32Type = mlir::IntegerType::get(ctx, 32);
    /// dims
    structElementTypes.push_back(mlir::LLVM::LLVMArrayType::get(ctx, i32Type,
      view.getShape().size()));
    /// meta pos
    structElementTypes.push_back(mlir::LLVM::LLVMPointerType::get(i32Type));
    structElementTypes.push_back(i32Type);
    /// meta crd
    structElementTypes.push_back(mlir::LLVM::LLVMPointerType::get(i32Type));
    structElementTypes.push_back(i32Type);
    /// data
    auto dataPtr = view.getElementType();
    if (auto complex = dyn_cast<mlir::ComplexType>(dataPtr)) {
      LLVMTypeConverter llvmConv(ctx);
      structElementTypes.push_back(
        mlir::LLVM::LLVMPointerType::get(llvmConv.convertType(complex)));
    }
    else {
      structElementTypes.push_back(
        mlir::LLVM::LLVMPointerType::get(dataPtr));
    }
    structElementTypes.push_back(i32Type);
    /// struct
    return mlir::LLVM::LLVMStructType::getLiteral(ctx, structElementTypes);
}

ViewTypeToPtrOfStructConverter::ViewTypeToPtrOfStructConverter() {
  addConversion([](Type type) { return type; });
  addConversion([&](ViewType view) -> Type {
    return convertView(view);
  });

  // Add generic source and target materializations to handle cases where
  // non-LLVM types persist after an LLVM conversion.
  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

mlir::Type ViewTypeToPtrOfStructConverter::convertView(ViewType view) {
  /// Using ptr to struct instead of struct to avoid having to worry
  /// abut ABI compatibility.
  ViewTypeToStructConverter structConv;
  auto strct = cast<mlir::LLVM::LLVMStructType>(structConv.convertView(view));
  return mlir::LLVM::LLVMPointerType::get(strct);
}

TensorToViewConverter::TensorToViewConverter() {
  addConversion([](Type type) { return type; });
  addConversion([&](mlir::RankedTensorType view) -> Type {
    return convertTensor(view);
  });
}

mlir::Type TensorToViewConverter::convertTensor(mlir::RankedTensorType tensor) {
  return ViewType::get(tensor.getContext(), tensor.getShape(),
    tensor.getElementType(), tensor.getEncoding());
}