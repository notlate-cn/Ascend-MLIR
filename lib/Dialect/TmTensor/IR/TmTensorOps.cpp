#include "Dialect/TmTensor/IR/TmTensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tm_tensor;

//===----------------------------------------------------------------------===//
// TmTensorDialect
//===----------------------------------------------------------------------===//

TmTensorDialect::TmTensorDialect(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<TmTensorDialect>()) {
  addOperations<AttentionOp>();
}

void TmTensorDialect::initialize() {}

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

void AttentionOp::build(OpBuilder &b, OperationState &state,
                        TypeRange resultTypes, ValueRange inputs,
                        ValueRange outputs) {
  state.addOperands(inputs);
  state.addOperands(outputs);
  state.addTypes(resultTypes);
  state.addAttribute("operand_segment_sizes",
                     b.getDenseI32ArrayAttr({(int32_t)inputs.size(),
                                            (int32_t)outputs.size()}));
}

// Parse: ins(ops : types) outs(ops : types) -> result_types
static ParseResult parseInsOuts(
    OpAsmParser &parser, StringRef keyword,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &types) {
  if (parser.parseKeyword(keyword) || parser.parseLParen())
    return failure();
  if (parser.parseOperandList(operands))
    return failure();
  if (parser.parseColon() || parser.parseTypeList(types))
    return failure();
  return parser.parseRParen();
}

ParseResult AttentionOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> insOperands, outsOperands;
  SmallVector<Type> insTypes, outsTypes;

  if (parseInsOuts(parser, "ins", insOperands, insTypes))
    return failure();
  if (parseInsOuts(parser, "outs", outsOperands, outsTypes))
    return failure();

  for (auto [op, ty] : llvm::zip(insOperands, insTypes))
    if (parser.resolveOperand(op, ty, result.operands))
      return failure();
  for (auto [op, ty] : llvm::zip(outsOperands, outsTypes))
    if (parser.resolveOperand(op, ty, result.operands))
      return failure();

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {(int32_t)insOperands.size(),
                           (int32_t)outsOperands.size()}));

  if (parser.parseArrow())
    return failure();
  SmallVector<Type> resultTypes;
  if (parser.parseTypeList(resultTypes))
    return failure();
  result.addTypes(resultTypes);
  return success();
}

void AttentionOp::print(OpAsmPrinter &p) {
  p << " ins(";
  p.printOperands(getInputs());
  p << " : ";
  llvm::interleaveComma(getInputs().getTypes(), p);
  p << ") outs(";
  p.printOperands(getOutputs());
  p << " : ";
  llvm::interleaveComma(getOutputs().getTypes(), p);
  p << ") -> ";
  llvm::interleaveComma(getOperation()->getResultTypes(), p);
}