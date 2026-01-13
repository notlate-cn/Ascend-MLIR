#ifndef AFIR_BASE
#define AFIR_BASE

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Core/Attrs.td"

//===----------------------------------------------------------------------===//
// 每个算子的公共属性
//===----------------------------------------------------------------------===//
class AFIR_CommonAttrs {
  dag commonArgs = (ins
    AffineMapArrayAttr:$indexing_maps,
    DefaultValuedOptionalAttr<I32Attr, "-1">:$loop_axis,
    DefaultValuedOptionalAttr<DictionaryAttr, "{}">:$ir_attr_def,
    OptionalAttr<TmpBufDescArrayAttr>:$tmp_buffers,
    AscTensorGroupsArrayAttr:$outputs
  );
}

//===----------------------------------------------------------------------===//
// Base AFIR Operation
//===----------------------------------------------------------------------===//
class AFIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<AFIR_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// 给定compute info的基类，作为所有后续算子的基类
//===----------------------------------------------------------------------===//
class AFIR_BaseOp<string mnemonic,
                          list<Trait> traits,
                          string computeType = "Invalid",
                          string computeUnit = "Invalid"> :
    AFIR_Op<mnemonic, traits> {
  code getComputeInfo = [{
    afir::ComputeType getComputeType() {
      return afir::ComputeType::}] # computeType # [{;
    }
    afir::ComputeUnit getComputeUnit() {
      return afir::ComputeUnit::}] # computeUnit # [{;
    }
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// N元操作：给定输入输出操作数的基类
//===----------------------------------------------------------------------===//
class AFIR_NaryOp<string mnemonic,
                          list<Trait> traits,
                          int numOperands,
                          int numResults = 1,
                          string computeType = "Invalid",
                          string computeUnit = "Invalid",
                          bit finalize = 0> :
    AFIR_BaseOp<mnemonic, traits, computeType, computeUnit> {
  code baseMethods = [{
      int getNumberOfOperands() { return }] # numOperands # [{; }
      int getNumberOfResults() { return }] # numResults # [{; }
    }] # getComputeInfo;
  let hasVerifier = 1;
  let extraClassDeclaration = !if(finalize, baseMethods, "");
}

//===----------------------------------------------------------------------===//
// 零元操作：输入数为0，输出数为1的基类
//===----------------------------------------------------------------------===//
class AFIR_NullaryOpBase<string mnemonic, 
                       list<Trait> traits,
                       list<Type> types,
                       string computeType = "Invalid",
                       string computeUnit = "Invalid",
                       bit finalize = 0> :
    AFIR_NaryOp<mnemonic, traits, 0, 1, computeType, computeUnit, finalize>, AFIR_CommonAttrs {
  let arguments = commonArgs;
  let results = (outs TensorOf<types>:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// 一元操作：输入数和输出数都为1的基类
//===----------------------------------------------------------------------===//
class AFIR_UnaryOpBase<string mnemonic, 
                       list<Trait> traits,
                       list<Type> inputTypes,
                       list<Type> resultTypes,
                       string computeType = "Invalid",
                       string computeUnit = "Invalid",
                       bit finalize = 0> :
    AFIR_NaryOp<mnemonic, traits, 1, 1, computeType, computeUnit, finalize>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins TensorOf<inputTypes>:$input),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

class AFIR_UnaryOp<string mnemonic, 
                    list<Trait> traits = [], 
                    list<Type> types, 
                    string computeType, 
                    string computeUnit,
                    bit finalize = 0> :
    AFIR_UnaryOpBase<mnemonic, traits, types, types, computeType, computeUnit, finalize>;

//===----------------------------------------------------------------------===//
// 二元操作：输入数为2和输出数为1的基类
//===----------------------------------------------------------------------===//
class AFIR_BinaryOpBase<string mnemonic, 
                        list<Trait> traits,
                        list<Type> lhsTypes,
                        list<Type> rhsTypes,
                        list<Type> resultTypes,
                        string computeType = "Invalid",
                        string computeUnit = "Invalid",
                        bit finalize = 0> :
    AFIR_NaryOp<mnemonic, traits, 2, 1, computeType, computeUnit, finalize>, AFIR_CommonAttrs {
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
  let hasVerifier = 1;
}

class AFIR_BinaryOp<string mnemonic, 
                    list<Trait> traits = [], 
                    list<Type> types, 
                    string computeType, 
                    string computeUnit,
                    bit finalize = 0> :
    AFIR_BinaryOpBase<mnemonic, traits, types, types, types, computeType, computeUnit, finalize>;

class AFIR_BinaryHomoOp<string mnemonic, 
                        list<Trait> traits = [], 
                        list<Type> inputTypes, 
                        list<Type> resultTypes, 
                        string computeType, 
                        string computeUnit,
                        bit finalize = 0> :
    AFIR_BinaryOpBase<mnemonic, traits, inputTypes, inputTypes, resultTypes, computeType, computeUnit, finalize>;

//===----------------------------------------------------------------------===//
// 三元操作：输入数为3输出数为1的基类
//===----------------------------------------------------------------------===//
class AFIR_TernaryOpBase<string mnemonic, 
                         list<Trait> traits,
                         list<Type> input1Types,
                         list<Type> input2Types,
                         list<Type> input3Types,
                         list<Type> resultTypes,
                         string computeType = "Invalid",
                         string computeUnit = "Invalid",
                         bit finalize = 0> :
    AFIR_NaryOp<mnemonic, traits, 3, 1, computeType, computeUnit, finalize>, AFIR_CommonAttrs {
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
  let hasVerifier = 1;
}

class AFIR_TernaryOp<string mnemonic, 
                    list<Trait> traits = [], 
                    list<Type> types, 
                    string computeType, 
                    string computeUnit,
                    bit finalize = 0> :
    AFIR_TernaryOpBase<mnemonic, traits, types, types, types, types, computeType, computeUnit, finalize>;

//===----------------------------------------------------------------------===//
// 可变输入数：输入数未知，输出数为1的基类
//===----------------------------------------------------------------------===//
class AFIR_VariadicOpBase<string mnemonic, 
                         list<Trait> traits,
                         list<Type> inputTypes,
                         list<Type> resultTypes,
                         string computeType = "Invalid",
                         string computeUnit = "Invalid",
                         bit finalize = 0> :
    AFIR_BaseOp<mnemonic, traits, computeType, computeUnit>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins Variadic<TensorOf<inputTypes>>:$inputs),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs) `->` type($result)
  }];
  code baseMethods = [{
      int getNumberOfOperands() { return getInputs().size(); }
      int getNumberOfResults() { return 1; }
    }] # getComputeInfo;

  let hasVerifier = 1;
  let extraClassDeclaration = !if(finalize, baseMethods, "");
}

class AFIR_VariadicOp<string mnemonic, 
                      list<Trait> traits = [], 
                      list<Type> types, 
                      string computeType, 
                      string computeUnit,
                      bit finalize = 0> :
    AFIR_VariadicOpBase<mnemonic, traits, types, types, computeType, computeUnit, finalize>;

#endif // AFIR_BASE