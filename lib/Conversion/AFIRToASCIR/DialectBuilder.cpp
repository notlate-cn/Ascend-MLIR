//===- DialectBuilder.cpp - ASC-IR dialect builder ----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file implements the dialect builder for the ASC-IR dialect.
// Follows the implementation pattern from onnx-mlir.
//
//===----------------------------------------------------------------------===//

#include "Conversion/AFIRToASCIR/DialectBuilder.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::afir;

namespace mlir {
namespace afir {

Value AscendCBuilder::createTBuf(ascendc::TPosition pos) const {
  auto bufferTy = ascendc::TBufType::get(b().getContext(), pos);
  return ascendc::TBufOp::create(b(), loc(), bufferTy);
}

Value AscendCBuilder::getTensorFromTBuf(Value tbuf, Type tensorType) const {
  return ascendc::TBufGetTensorOp::create(b(), loc(), tensorType, tbuf);
}

Value AscendCBuilder::addL3(Value dst, Value lhs, Value rhs) const {
  ascendc::AddL3Op::create(b(), loc(), dst, lhs, rhs);
  return dst;
}

Value AscendCBuilder::subL3(Value dst, Value lhs, Value rhs) const {
  ascendc::SubL3Op::create(b(), loc(), dst, lhs, rhs);
  return dst;
}

Value AscendCBuilder::mulL3(Value dst, Value lhs, Value rhs) const {
  ascendc::MulL3Op::create(b(), loc(), dst, lhs, rhs);
  return dst;
}

Value AscendCBuilder::divL3(Value dst, Value lhs, Value rhs) const {
  ascendc::DivL3Op::create(b(), loc(), dst, lhs, rhs);
  return dst;
}

Value AscendCBuilder::createLocalTensor(Type tensorType, Value tbuf) const {
  return ascendc::TBufGetTensorOp::create(b(), loc(), tensorType, tbuf);
}

} // namespace afir
} // namespace mlir
