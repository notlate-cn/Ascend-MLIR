//===- Base.td - AFIR dialect base definitions --------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the base structures for the AFIR dialect type system,
// including the dialect definition and type/attribute base classes.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CORE_BASE
#define AFIR_CORE_BASE

include "Dialect/AFIR/Dialect.td"

//===----------------------------------------------------------------------===//
// Base AFIR Type Definition
//===----------------------------------------------------------------------===//

class AFIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AFIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Base AFIR Attribute Definition
//===----------------------------------------------------------------------===//

class AFIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AFIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // AFIR_CORE_BASE