//===-- QuiccirTypes.td - Quiccir types definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUICCIR_QUICCIRTYPES
#define QUICCIR_QUICCIRTYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "Quiccir/QuiccirDialect.td"

//===----------------------------------------------------------------------===//
// Base quiccir type definition.
//===----------------------------------------------------------------------===//

class Quiccir_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Quiccir_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ViewType
//===----------------------------------------------------------------------===//

def Quiccir_ViewType : Quiccir_Type<"View", "view", [ShapedTypeInterface]> {
  let summary = "3D bufferized tensor";
  let description = [{
    Bufferized sparse tensor designed to map to either QuICC::View for direct library calls
//  (via !llvm.struct) or to MLIR sparse tensor (via memRef triplets)
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    "Attribute":$encoding
    // memorySpace attribute?
  );

  let extraClassDeclaration = [{
    /// Returns if this type is ranked (always true).
    bool hasRank() const { return true; }

    /// Clone view.
    ViewType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                         Type elementType) const;
  }];

  // let skipDefaultBuilders = 1;
  // let genVerifyDecl = 1;
  // let genStorageClass = 0;

  let hasCustomAssemblyFormat = 1;
}

// Whether a type is a ViewType.
def IsViewTypePred : CPred<"::llvm::isa<::mlir::quiccir::ViewType>($_self)">;

// Any view type whose element type is from the given `allowedTypes`
// list, and which additionally satisfies an optional list of predicates.
class ViewOf<
    list<Type> allowedTypes,
    list<Pred> preds = [],
    string summary = "view">
  : ShapedContainerType<allowedTypes,
      And<!listconcat([IsViewTypePred], preds)>,
      summary, "::mlir::quiccir::ViewType">;

def AnyView  : ViewOf<[AnyType]>;

#endif // QUICCIR_QUICCIRTYPES
