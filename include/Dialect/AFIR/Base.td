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
// Base AFIR Operation
//===----------------------------------------------------------------------===//
class AFIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<AFIR_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Base AFIR Operation with Counting and Compute Info
//===----------------------------------------------------------------------===//
class AFIR_BaseOp<string mnemonic,
                          list<Trait> traits,
                          int numOperands,
                          int numResults = 1,
                          string computeType = "Invalid",
                          string computeUnit = "Invalid"> :
    AFIR_Op<mnemonic, traits> {
  let extraClassDeclaration = [{
    static int getNumberOfOperands() { return }] # numOperands # [{; }
    static int getNumberOfResults() { return }] # numResults # [{; }
    static afir::ComputeType getComputeType() {
      return afir::ComputeType::}] # computeType # [{;
    }
    static afir::ComputeUnit getComputeUnit() {
      return afir::ComputeUnit::}] # computeUnit # [{;
    }
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Unary Operations (从通用基类派生)
//===----------------------------------------------------------------------===//
class AFIR_UnaryOpBase<string mnemonic, 
                       list<Trait> traits,
                       list<Type> inputTypes,
                       list<Type> resultTypes,
                       string computeType = "Invalid",
                       string computeUnit = "Invalid"> :
    AFIR_BaseOp<mnemonic, traits, 1, 1, computeType, computeUnit>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins TensorOf<inputTypes>:$input),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

class AFIR_UnaryOp<string mnemonic, 
                    list<Trait> traits = [], 
                    list<Type> types, 
                    string computeType, 
                    string computeUnit> :
    AFIR_UnaryOpBase<mnemonic, traits, types, types, computeType, computeUnit>;

//===----------------------------------------------------------------------===//
// Binary Operations (从通用基类派生)
//===----------------------------------------------------------------------===//
class AFIR_BinaryOpBase<string mnemonic, 
                        list<Trait> traits,
                        list<Type> lhsTypes,
                        list<Type> rhsTypes,
                        list<Type> resultTypes,
                        string computeType = "Invalid",
                        string computeUnit = "Invalid"> :
    AFIR_BaseOp<mnemonic, traits, 2, 1, computeType, computeUnit>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins 
      TensorOf<lhsTypes>:$lhs, 
      TensorOf<rhsTypes>:$rhs
    ),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

class AFIR_BinaryOp<string mnemonic, 
                    list<Trait> traits = [], 
                    list<Type> types, 
                    string computeType, 
                    string computeUnit> :
    AFIR_BinaryOpBase<mnemonic, traits, types, types, types, computeType, computeUnit>;

class AFIR_BinaryHomoOp<string mnemonic, 
                        list<Trait> traits = [], 
                        list<Type> inputTypes, 
                        list<Type> resultTypes, 
                        string computeType, 
                        string computeUnit> :
    AFIR_BinaryOpBase<mnemonic, traits, inputTypes, inputTypes, resultTypes, computeType, computeUnit>;

//===----------------------------------------------------------------------===//
// Ternary Operations (从通用基类派生)
//===----------------------------------------------------------------------===//
class AFIR_TernaryOpBase<string mnemonic, 
                         list<Trait> traits,
                         list<Type> input1Types,
                         list<Type> input2Types,
                         list<Type> input3Types,
                         list<Type> resultTypes,
                         string computeType = "Invalid",
                         string computeUnit = "Invalid"> :
    AFIR_BaseOp<mnemonic, traits, 3, 1, computeType, computeUnit>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins 
      TensorOf<input1Types>:$input1,
      TensorOf<input2Types>:$input2,
      TensorOf<input3Types>:$input3
    ),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $input1 `,` $input2 `,` $input3 attr-dict `:` 
    `(` type($input1) `,` type($input2) `,` type($input3) `)` `->` type($result)
  }];
}

class AFIR_TernaryConditionOp<string mnemonic, 
                              list<Trait> traits = [], 
                              list<Type> conditionTypes,
                              list<Type> types, 
                              string computeType, 
                              string computeUnit> :
    AFIR_TernaryOpBase<mnemonic, traits, conditionTypes, types, types, types, computeType, computeUnit>;

#endif // AFIR_BASE