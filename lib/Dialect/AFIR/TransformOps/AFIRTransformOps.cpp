//===- AFIRTransformOps.cpp - Implementation of AFIR transform ops ------===//
//
// Part of Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/TransformOps/AFIRTransformOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AddIndexArgsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::AddIndexArgsOp::apply(transform::TransformRewriter &rewriter,
                                transform::TransformResults &results,
                                transform::TransformState &state) {
  auto payloadOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payloadOps))
    return emitDefiniteFailure() << "requires a single function to operate on";

  auto funcOp = dyn_cast<func::FuncOp>(*payloadOps.begin());
  if (!funcOp)
    return emitSilenceableFailure(getLoc())
           << "target is expected to be a func.func operation";

  int64_t numArgs = getNumArgs();
  if (numArgs <= 0)
    return emitSilenceableFailure(getLoc())
           << "number of arguments to add must be positive";

  if (!funcOp.getBody().hasOneBlock())
    return emitDefiniteFailure() << "expected function to have exactly one block";

  SmallVector<Type> newArgTypes = llvm::to_vector(funcOp.getArgumentTypes());
  SmallVector<DictionaryAttr> newArgAttrs;
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    newArgAttrs.push_back(funcOp.getArgAttrDict(i));
  }

  SmallVector<Value> newArgs;
  newArgs.reserve(numArgs);

  for (int64_t i = 0; i < numArgs; ++i) {
    newArgTypes.push_back(IntegerType::get(funcOp.getContext(), 64));
    newArgAttrs.push_back(DictionaryAttr::get(funcOp.getContext()));
  }

  FunctionType newFuncType = FunctionType::get(
      funcOp.getContext(), newArgTypes, funcOp.getResultTypes());

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(funcOp);
  auto newFuncOp = rewriter.create<func::FuncOp>(
      funcOp.getLoc(), funcOp.getName(), newFuncType);
  newFuncOp.setVisibility(funcOp.getVisibility());
  newFuncOp->setDiscardableAttrs(funcOp->getDiscardableAttrDictionary());

  Region &newRegion = newFuncOp.getBody();
  rewriter.createBlock(&newRegion, newRegion.begin(), newArgTypes, 
                     SmallVector<Location>(newArgTypes.size(), funcOp.getLoc()));

  for (unsigned i = 0; i < newArgTypes.size(); ++i) {
    newFuncOp.setArgAttrs(i, newArgAttrs[i]);
  }

  IRMapping operandMapper;
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    operandMapper.map(funcOp.getArgument(i), newFuncOp.getArgument(i));
  }

  for (int64_t i = 0; i < numArgs; ++i) {
    unsigned argIndex = newArgTypes.size() - numArgs + i;
    newArgs.push_back(newFuncOp.getArgument(argIndex));
  }

  rewriter.setInsertionPointToStart(&newFuncOp.getBody().front());
  for (Operation &op : funcOp.getOps())
    rewriter.clone(op, operandMapper);

  SmallVector<Operation *> argOps;
  for (int64_t i = 0; i < numArgs; ++i) {
    unsigned argIndex = newArgTypes.size() - numArgs + i;
    Value arg = newFuncOp.getArgument(argIndex);
    rewriter.setInsertionPointToStart(&newFuncOp.getBody().front());
    auto argOp = rewriter.create<arith::IndexCastOp>(
        newFuncOp.getLoc(), IndexType::get(newFuncOp.getContext()), arg);
    argOps.push_back(argOp);
  }

  rewriter.eraseOp(funcOp);

  results.set(cast<OpResult>(getTransformed()), {newFuncOp});
  for (int64_t i = 0; i < numArgs; ++i) {
    results.set(cast<OpResult>(getNewArgs()[i]), {argOps[i]});
  }

  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::AddIndexArgsOp::verify() {
  if (getNumArgs() <= 0)
    return emitOpError() << "number of arguments to add must be positive";

  int64_t numArgs = getNumArgs();
  if (static_cast<int64_t>(getNewArgs().size()) != numArgs)
    return emitOpError() << "number of new args results (" << getNewArgs().size()
                        << ") does not match num_args (" << numArgs << ")";

  return success();
}

void transform::AddIndexArgsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class AFIRTransformDialectExtension
    : public transform::TransformDialectExtension<
          AFIRTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRTransformDialectExtension)

  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "Dialect/AFIR/TransformOps/AFIRTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "Dialect/AFIR/TransformOps/AFIRTransformOps.cpp.inc"

void mlir::afir::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<AFIRTransformDialectExtension>();
}
