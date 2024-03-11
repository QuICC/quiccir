//===- Utils.h - Quiccir Lowering to Call Utils -----------------*- C++ -*-===//
//
// This header file defines prototypes of Quiccir Lowering to Call Utils.
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_TRANSFORMS_UTILS_H
#define QUICCIR_TRANSFORMS_UTILS_H

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace quiccir {

//===----------------------------------------------------------------------===//
// Quiccir Lower to call Utils
//===----------------------------------------------------------------------===//

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
  if (isa<AllocOp>(op)) {
    // replace with getProducer
    for (const NamedAttribute att : op->getAttrs()) {
      if (auto as = att.getValue().dyn_cast<StringAttr>()) {
        std::string astr = as.str();
        auto pos = astr.find("quiccir.");
        astr.erase(pos, 8);
        fnName += "_"+astr;
      }
    }
  }
  // Return types
  for (Type ret : op->getResultTypes()) {
    if (auto tensor = ret.dyn_cast<RankedTensorType>()) {
      auto as = tensor.getEncoding().cast<StringAttr>();
      fnName += "_"+as.str();
    }
    if (auto view = ret.dyn_cast<ViewType>()) {
      auto as = view.getEncoding().cast<StringAttr>();
      fnName += "_"+as.str();
    }
  }
  // Argument types
  for (Type arg : op->getOperandTypes()) {
    if (auto tensor = arg.dyn_cast<RankedTensorType>()) {
      auto as = tensor.getEncoding().cast<StringAttr>();
      fnName += "_"+as.str();
    }
    if (auto view = arg.dyn_cast<ViewType>()) {
      auto as = view.getEncoding().cast<StringAttr>();
      fnName += "_"+as.str();
    }
  }
  std::replace(fnName.begin(), fnName.end(), '.', '_');
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