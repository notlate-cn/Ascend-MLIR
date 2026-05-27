//===- NetworkJsonEmitter.h - Emit network.json for a coordinator func ----===//
//
// Walks a network func body's call ops, classifies each callee as either an
// AscendC kernel group or an aclnn op (presence of "aclnn.op" attr), and writes
// network.json per docs/superpowers/specs/2026-05-13-network-runner-mixed-cpu-sim-design.md §3.1.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::auto_fuse {

// Writes JSON describing the network coordinator's call graph to `os`.
// `coord` is the public (non-private) func; `module` is its enclosing module
// (used to look up callees). Returns an error if the body contains an
// unsupported op other than func.call / func.return / tensor.cast.
llvm::Error emitNetworkJson(mlir::ModuleOp module,
                            mlir::func::FuncOp coord,
                            llvm::raw_ostream &os);

// Writes a debug-only provenance sidecar (see docs/auto-fuse/debug.md §7.5).
// For each callee in the coordinator body, records the source ops fused into
// it (id / name / op_role / loc / result_ssa), a kernel-level fused_ops_summary
// joined from op_roles in topological order, and which source op produces each
// kernel result (boundary_source_ops). aclnn kernels report a single synthetic
// source op with role = the @__aclnn_<...> op name. The output schema is
// independent of network.json's; the two files relate via `kernel_id` only.
//
// PROVENANCE IS DEBUG-ONLY. The main compilation pipeline must not consume it.
llvm::Error emitNetworkProvenanceJson(mlir::ModuleOp module,
                                      mlir::func::FuncOp coord,
                                      llvm::raw_ostream &os);

} // namespace mlir::auto_fuse
