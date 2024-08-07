//===- Utils.cpp - Quiccir transform utils ---------------------*- C++ -*-===//


#include "Quiccir/Transforms/Utils.h"

namespace mlir {
namespace quiccir {

std::string perm2str(Operation* op) {
  std::string permStr;
  if (auto traOp = dyn_cast<TransposeOp>(op)) {
    auto perm = traOp.getPermutation();
    permStr += "_";
    for (auto&& i : perm) {
      permStr += std::to_string(i);
    }
  }
  return permStr;
}

std::string kind2str(Operation* op) {
  if (auto trOp = dyn_cast<TransformOpInterface>(op)) {
    return trOp.getKind();
  }
  return std::string{};
}

} // namespace quiccir
} // namespace mlir
