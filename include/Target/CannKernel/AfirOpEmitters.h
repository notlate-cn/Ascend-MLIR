//===- AfirOpEmitters.h - Custom emitters for missing PyAsc ops --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// Provides printOperation overloads for AscendC dialect ops that are not yet
// implemented in PyAsc's Translation.cpp / PrintableOpTypes list.
//
// Currently handled:
//   ascendc::BroadcastL2Op      -- uses adv_api Broadcast<T,N,axis>
//   ascendc::ReduceSum2DL2Op    -- uses adv_api ReduceSum<T,AR/RA>
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_TARGET_CANNKERNEL_AFIROPEMITTERS_H
#define AFIR_TARGET_CANNKERNEL_AFIROPEMITTERS_H

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Target/Asc/CodeEmitter.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::afir {

/// Try to emit `op` using the AFIR-local emitters.
/// Returns success if `op` was handled, failure if the op is not one of the
/// locally-handled ops and the caller should fall back to PyAsc emitOperation.
mlir::LogicalResult tryEmitAfirOp(CodeEmitter &emitter, mlir::Operation &op);

} // namespace mlir::afir

#endif // AFIR_TARGET_CANNKERNEL_AFIROPEMITTERS_H
