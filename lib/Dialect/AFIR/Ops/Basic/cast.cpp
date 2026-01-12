//===- cast.cpp - AFIR cast operation implementations -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::afir;

//===----------------------------------------------------------------------===//
// Type ID 枚举定义
//===----------------------------------------------------------------------===//
enum class CastTypeId : unsigned {
  F32 = 0,
  F16 = 1,
  I64 = 2,
  I32 = 3,
  I16 = 4,
  BF16 = 5,
  I8 = 6,
  UI8 = 7,
  I4 = 8,
  UI32 = 9,
  UI16 = 10,
  UI64 = 11,
  Unknown = 999
};

//===----------------------------------------------------------------------===//
// 类型到 ID 的映射
//===----------------------------------------------------------------------===//
static CastTypeId getTypeId(Type type) {
  // 浮点类型
  if (auto floatTy = llvm::dyn_cast<FloatType>(type)) {
    if (floatTy.isF32()) return CastTypeId::F32;
    if (floatTy.isF16()) return CastTypeId::F16;
    if (floatTy.isBF16()) return CastTypeId::BF16;
    return CastTypeId::Unknown;
  }

  // 整数类型
  if (auto intTy = llvm::dyn_cast<IntegerType>(type)) {
    unsigned width = intTy.getWidth();
    bool isUnsigned = intTy.isUnsigned();
    bool isSignless = intTy.isSignless();

    switch (width) {
      case 64:
        if (isUnsigned) return CastTypeId::UI64;
        if (isSignless || intTy.isSigned()) return CastTypeId::I64;
        break;
      case 32:
        if (isUnsigned) return CastTypeId::UI32;
        if (isSignless || intTy.isSigned()) return CastTypeId::I32;
        break;
      case 16:
        if (isUnsigned) return CastTypeId::UI16;
        if (isSignless || intTy.isSigned()) return CastTypeId::I16;
        break;
      case 8:
        if (isUnsigned) return CastTypeId::UI8;
        if (isSignless || intTy.isSigned()) return CastTypeId::I8;
        break;
      case 4:
        return CastTypeId::I4;
    }
  }

  return CastTypeId::Unknown;
}

//===----------------------------------------------------------------------===//
// 支持的类型转换对查找表
//===----------------------------------------------------------------------===//
static const llvm::DenseSet<std::pair<CastTypeId, CastTypeId>> &getSupportedCastPairs() {
  using CT = CastTypeId;

  static const llvm::DenseSet<std::pair<CT, CT>> supportedCasts = {// F32 可以转换为: F32, F16, I64, I32, I16, BF16
                                                                   {CT::F32, CT::F32},
                                                                   {CT::F32, CT::F16},
                                                                   {CT::F32, CT::I64},
                                                                   {CT::F32, CT::I32},
                                                                   {CT::F32, CT::I16},
                                                                   {CT::F32, CT::BF16},

                                                                   // F16 可以转换为: F32, I32, I16, I8, UI8, I4, I64
                                                                   {CT::F16, CT::F32},
                                                                   {CT::F16, CT::I32},
                                                                   {CT::F16, CT::I16},
                                                                   {CT::F16, CT::I8},
                                                                   {CT::F16, CT::UI8},
                                                                   {CT::F16, CT::I4},
                                                                   {CT::F16, CT::I64},

                                                                   // I4 可以转换为: F16
                                                                   {CT::I4, CT::F16},

                                                                   // UI8 可以转换为: F16, F32, I32, I16, I8, I4
                                                                   {CT::UI8, CT::F16},
                                                                   {CT::UI8, CT::F32},
                                                                   {CT::UI8, CT::I32},
                                                                   {CT::UI8, CT::I16},
                                                                   {CT::UI8, CT::I8},
                                                                   {CT::UI8, CT::I4},

                                                                   // I8 可以转换为: F16, UI8
                                                                   {CT::I8, CT::F16},
                                                                   {CT::I8, CT::UI8},

                                                                   // I16 可以转换为: F16, F32, UI16
                                                                   {CT::I16, CT::F16},
                                                                   {CT::I16, CT::F32},
                                                                   {CT::I16, CT::UI16},

                                                                   // I32 可以转换为: F32, I64, I16, F16, UI32
                                                                   {CT::I32, CT::F32},
                                                                   {CT::I32, CT::I64},
                                                                   {CT::I32, CT::I16},
                                                                   {CT::I32, CT::F16},
                                                                   {CT::I32, CT::UI32},

                                                                   // I64 可以转换为: I32
                                                                   {CT::I64, CT::I32},

                                                                   // BF16 可以转换为: F32, I32
                                                                   {CT::BF16, CT::F32},
                                                                   {CT::BF16, CT::I32},

                                                                   // UI32 可以转换为: I32
                                                                   {CT::UI32, CT::I32},

                                                                   // UI16 可以转换为: I16
                                                                   {CT::UI16, CT::I16},

                                                                   // UI64 可以转换为: I64
                                                                   {CT::UI64, CT::I64}};

  return supportedCasts;
}

//===----------------------------------------------------------------------===//
// 检查类型转换是否被支持
//===----------------------------------------------------------------------===//
static bool isSupportedCast(Type inputType, Type outputType) {
  CastTypeId inputId = getTypeId(inputType);
  CastTypeId outputId = getTypeId(outputType);

  if (inputId == CastTypeId::Unknown || outputId == CastTypeId::Unknown) {
    return false;
  }

  return getSupportedCastPairs().count({inputId, outputId}) > 0;
}

//===----------------------------------------------------------------------===//
// CastOp Verify 实现
//===----------------------------------------------------------------------===//
LogicalResult CastOp::verify() {
  // 1. 检查输入输出都是 ranked tensor
  auto inputTensorType = mlir::dyn_cast<RankedTensorType>(getInput().getType());
  auto outputTensorType = mlir::dyn_cast<RankedTensorType>(getResult().getType());

  if (!inputTensorType || !outputTensorType) {
    return emitOpError("input and output must be ranked tensors");
  }

  // 2. 检查 shape 必须完全匹配
  if (inputTensorType.getShape() != outputTensorType.getShape()) {
    return emitOpError("input and output shapes must match, got input shape ")
           << inputTensorType.getShape() << " and output shape " << outputTensorType.getShape();
  }

  // 3. 检查元素类型转换是否被支持
  Type inputElemType = inputTensorType.getElementType();
  Type outputElemType = outputTensorType.getElementType();

  if (!isSupportedCast(inputElemType, outputElemType)) {
    return emitOpError("unsupported cast from ")
           << inputElemType << " to " << outputElemType << ". See supported cast pairs in Cast operation definition.";
  }

  return success();
}
