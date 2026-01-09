//===- Base.td - AFIR dialect base definitions --------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the base structures for AFIR operations, including
// operation base classes and common templates.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_BASE
#define AFIR_BASE

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Core/Attrs.td"

//===----------------------------------------------------------------------===//
// Common Attributes Mixin
//===----------------------------------------------------------------------===//

class AFIR_CommonAttrs {
  dag commonArgs = (ins
    AffineMapArrayAttr:$indexing_maps,
    DefaultValuedOptionalAttr<I32Attr, "-1">:$loop_axis,
    DefaultValuedOptionalAttr<DictionaryAttr, "{}">:$ir_attr_def,
    OptionalAttr<TmpBufDescArrayAttr>:$tmp_buffers,
    OptionalAttr<AscTensorGroupsArrayAttr>:$outputs
  );
}

//===----------------------------------------------------------------------===//
// Base AFIR Operation Definition
//===----------------------------------------------------------------------===//

class AFIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<AFIR_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Base AFIR Binary Operation
//===----------------------------------------------------------------------===//

class AFIR_BinaryOp<string mnemonic, list<Trait> traits = [], list<Type> types> :
    AFIR_Op<mnemonic, traits>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins
      TensorOf<types>:$lhs,
      TensorOf<types>:$rhs
    ),
    commonArgs
  );

  let results = (outs TensorOf<types>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 2;
    }
    static int getNumberOfResults() {
      return 1;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Base AFIR Unary Operation
//===----------------------------------------------------------------------===//

class AFIR_UnaryOp<string mnemonic, list<Trait> traits = [], list<Type> types> :
    AFIR_Op<mnemonic, traits>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins
      TensorOf<types>:$input
    ),
    commonArgs
  );

  let results = (outs TensorOf<types>:$result);

  let assemblyFormat = [{
    $input attr-dict `:` `(` type($input) `)` `->` type($result)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 1;
    }
    static int getNumberOfResults() {
      return 1;
    }
  }];
}

#endif // AFIR_BASE