//===- AFIRDialect.td - AFIR dialect definition ------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the AFIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_DIALECT
#define AFIR_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// AFIR Dialect Definition
//===----------------------------------------------------------------------===//

def AFIR_Dialect : Dialect {
  let name = "afir";
  let summary = "Ascend Frontend IR Dialect";
  let description = [{
    AFIR (Ascend Frontend IR) is a high-level dialect designed for
    representing computational operations targeting Ascend hardware.
    It provides a bridge between high-level ML frameworks and lower-level
    hardware-specific dialects like ASC-IR.
  }];
  let cppNamespace = "::mlir::afir";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base AFIR Operation Definition
//===----------------------------------------------------------------------===//

class AFIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<AFIR_Dialect, mnemonic, traits>;

#endif // AFIR_DIALECT
