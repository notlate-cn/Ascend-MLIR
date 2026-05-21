#ifndef ASCEND_MLIR_DIALECT_TMTENSOR_IR_TMTENSOR_H
#define ASCEND_MLIR_DIALECT_TMTENSOR_IR_TMTENSOR_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::tm_tensor {

//===----------------------------------------------------------------------===//
// TmTensorDialect
//===----------------------------------------------------------------------===//

class TmTensorDialect : public Dialect {
public:
  static StringRef getDialectNamespace() { return "tm_tensor"; }
  explicit TmTensorDialect(MLIRContext *ctx);
  void initialize();
};

//===----------------------------------------------------------------------===//
// AttentionOp — minimal stub for tm_tensor.attention
//
// Syntax:
//   tm_tensor.attention
//       ins(%q, %k, %v, %scale : T1, T2, T3, T4)
//       outs(%init : T5)
//       -> T5
//
// Semantics: scaled dot-product attention (opaque; lowered to linalg by
// ConvertTmTensorAttentionToLinalg before entering the core pipeline).
//===----------------------------------------------------------------------===//

class AttentionOp
    : public Op<AttentionOp, OpTrait::AtLeastNResults<1>::Impl,
                MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "tm_tensor.attention"; }

  static void build(OpBuilder &b, OperationState &state, TypeRange resultTypes,
                    ValueRange inputs, ValueRange outputs);

  // ins operands (query, key, value, scale)
  OperandRange getInputs() {
    auto sizes = (*this)->getAttrOfType<DenseI32ArrayAttr>(
        "operand_segment_sizes");
    return getOperation()->getOperands().slice(0, sizes[0]);
  }
  // outs operands (init tensors)
  OperandRange getOutputs() {
    auto sizes = (*this)->getAttrOfType<DenseI32ArrayAttr>(
        "operand_segment_sizes");
    return getOperation()->getOperands().slice(sizes[0], sizes[1]);
  }

  // Required by mlir::Op machinery (not ODS-generated here).
  static ArrayRef<StringRef> getAttributeNames() {
    static constexpr StringRef names[] = {"operand_segment_sizes"};
    return ArrayRef<StringRef>(names);
  }

  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);

  // MemoryEffectOpInterface: pure (no side effects on tensors in SSA form)
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {}
};

} // namespace mlir::tm_tensor

#endif