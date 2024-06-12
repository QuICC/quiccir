//===- Utils.h - Quiccir Lowering to Call Utils -----------------*- C++ -*-===//
//
// This header file defines prototypes of Quiccir Lowering to Call Utils.
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_TRANSFORMS_UTILS_H
#define QUICCIR_TRANSFORMS_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quiccir/IR/QuiccirOps.h"

namespace mlir {
namespace quiccir {

//===----------------------------------------------------------------------===//
// Quiccir Lower to call Utils
//===----------------------------------------------------------------------===//

/// return permutation as a string
std::string perm2str(Operation* op);

/// @brief Get a SymbolRefAttr containing the library function name for the op.
/// If the library function does not exist, insert a declaration.
/// @tparam OpT
/// @param op
/// @param rewriter
/// @param types
/// @return
template <class OpT>
static FailureOr<FlatSymbolRefAttr>
getLibraryCallSymbolRef(Operation *op, PatternRewriter &rewriter, ArrayRef<Type> types) {
  // lib call name mangling
  auto implOp = cast<OpT>(op);
  std::string fnName = implOp.getOperationName().str();
  // Attributes for alloc op
  if (auto allocOp = dyn_cast<AllocOp>(op)) {
    std::string astr = allocOp.getProducer().str();
    auto pos = astr.find("quiccir.");
    astr.erase(pos, 8);
    fnName += "_"+astr;
  }
  // Attribute for transpose op
  fnName += perm2str(op);
  // Return types
  /// \todo remove code dupliation
  for (Type ret : op->getResultTypes()) {
    if (auto tensor = ret.dyn_cast<RankedTensorType>()) {
      auto eleTy = tensor.getElementType();
      std::string tyStr;
      llvm::raw_string_ostream tyOS(tyStr);
      eleTy.print(tyOS);
      if (isa<ComplexType>(eleTy)) {
        tyStr.erase(std::find(tyStr.begin(), tyStr.end(), '<'));
        tyStr.erase(std::find(tyStr.begin(), tyStr.end(), '>'));
      }
      fnName += "_"+tyStr;
      if (!tensor.getEncoding()) {
        return rewriter.notifyMatchFailure(op, "Encoding attribute is missing");
      }
      auto as = tensor.getEncoding().cast<StringAttr>();
      fnName += "_"+as.str();
    }
    if (auto view = ret.dyn_cast<ViewType>()) {
      auto eleTy = view.getElementType();
      std::string tyStr;
      llvm::raw_string_ostream tyOS(tyStr);
      eleTy.print(tyOS);
      if (isa<ComplexType>(eleTy)) {
        tyStr.erase(std::find(tyStr.begin(), tyStr.end(), '<'));
        tyStr.erase(std::find(tyStr.begin(), tyStr.end(), '>'));
      }
      fnName += "_"+tyStr;
      auto as = view.getEncoding().cast<StringAttr>();
      fnName += "_"+as.str();
    }
    if (auto memRef = ret.dyn_cast<MemRefType>()) {
      auto eleTy = memRef.getElementType();
      std::string tyStr;
      llvm::raw_string_ostream tyOS(tyStr);
      eleTy.print(tyOS);
      if (isa<ComplexType>(eleTy)) {
        tyStr.erase(std::find(tyStr.begin(), tyStr.end(), '<'));
        tyStr.erase(std::find(tyStr.begin(), tyStr.end(), '>'));
      }
      fnName += "_"+tyStr;
    }
  }
  // Argument types
  for (Type arg : op->getOperandTypes()) {
    if (auto tensor = arg.dyn_cast<RankedTensorType>()) {
      auto eleTy = tensor.getElementType();
      std::string tyStr;
      llvm::raw_string_ostream tyOS(tyStr);
      eleTy.print(tyOS);
      if (isa<ComplexType>(eleTy)) {
        tyStr.erase(std::find(tyStr.begin(), tyStr.end(), '<'));
        tyStr.erase(std::find(tyStr.begin(), tyStr.end(), '>'));
      }
      fnName += "_"+tyStr;
      if (!tensor.getEncoding()) {
        return rewriter.notifyMatchFailure(op, "Encoding attribute is missing");
      }
      auto as = tensor.getEncoding().cast<StringAttr>();
      fnName += "_"+as.str();
    }
    if (auto view = arg.dyn_cast<ViewType>()) {
      auto eleTy = view.getElementType();
      std::string tyStr;
      llvm::raw_string_ostream tyOS(tyStr);
      eleTy.print(tyOS);
      if (isa<ComplexType>(eleTy)) {
        tyStr.erase(std::find(tyStr.begin(), tyStr.end(), '<'));
        tyStr.erase(std::find(tyStr.begin(), tyStr.end(), '>'));
      }
      fnName += "_"+tyStr;
      auto as = view.getEncoding().cast<StringAttr>();
      fnName += "_"+as.str();
    }
    if (auto memRef = arg.dyn_cast<MemRefType>()) {
      auto eleTy = memRef.getElementType();
      std::string tyStr;
      llvm::raw_string_ostream tyOS(tyStr);
      eleTy.print(tyOS);
      if (isa<ComplexType>(eleTy)) {
        tyStr.erase(std::find(tyStr.begin(), tyStr.end(), '<'));
        tyStr.erase(std::find(tyStr.begin(), tyStr.end(), '>'));
      }
      fnName += "_"+tyStr;
    }
  }
  std::replace(fnName.begin(), fnName.end(), '.', '_');

  // Layout attr
  if (auto allocDataOp = dyn_cast<AllocDataOp>(op)) {
    fnName += "_"+allocDataOp.getLayoutAttrName().str();
  }

  if (fnName.empty())
    return rewriter.notifyMatchFailure(op, "No library call defined for: ");
  fnName = "_ciface_"+fnName;

  // fnName is a dynamic std::string, unique it via a SymbolRefAttr.
  FlatSymbolRefAttr fnNameAttr =
      SymbolRefAttr::get(rewriter.getContext(), fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnNameAttr.getAttr()))
    return fnNameAttr;

  auto libFnType = rewriter.getFunctionType(types, {});

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  func::FuncOp funcOp = rewriter.create<func::FuncOp>(
      op->getLoc(), fnNameAttr.getValue(), libFnType);
  funcOp.setPrivate();

  /// \todo set noalias attribute
  // SmallVector<Attribute, 4> argAttrs;
  // for ([[maybe_unused]] auto t : types) {
  //   Attribute at = StringAttr::get(rewriter.getContext(), LLVM::LLVMDialect::getNoAliasAttrName());
  //   argAttrs.push_back(at);
  // }
  // funcOp.setAllArgAttrs(argAttrs);
  return fnNameAttr;
}


} // namespace quiccir
} // namespace mlir

#endif // QUICCIR_TRANSFORMS_UTILS_H