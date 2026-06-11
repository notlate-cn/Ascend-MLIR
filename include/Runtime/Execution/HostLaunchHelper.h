//===- HostLaunchHelper.h - One-shot AscendC kernel launch from generated code -===//
//
// Used by the network_host.cpp emitted by AclnnBackend. Internally driven by
// NativeExecutionRunner in Simulation mode (libruntime_camodel.so), so the
// generated host can launch AscendC kernels on the CANN CPU sim without
// requiring aclInit.
//
// Thread-safety: process-wide singletons are lazy-initialized under a mutex.
//===----------------------------------------------------------------------===//
#pragma once

#include "Runtime/AclnnOps.h"  // for TensorInfo

namespace mlir::runtime {

// Launch AscendC kernel `kernelName` once.
//
// `kernelBinariesDir` should contain `<kernelName>/<kernelName>.o` (matches
// runtime-session --kernel ... --output ... --name ... layout).
// `tilingsPath` is the JSON written by the runner:
//   {kernelName: {param: value, ..., "_block_dim": N}, ...}
//
// `inputs`/`outputs` are TensorInfos with host pointers, shape, dtype set
// (outputs.data must be a host buffer of the right size before calling).
// Returns 0 on success; non-zero on any underlying error (diagnostic to stderr).
extern "C" int hostLaunchAscendCKernel(
    const char *kernelName,
    const char *kernelBinariesDir,
    const char *tilingsPath,
    aclnn::TensorInfo *inputs, int numInputs,
    aclnn::TensorInfo *outputs, int numOutputs);

// Optional dump dir for per-kernel intermediates; pass nullptr to disable.
// Per-kernel npys are written as DIR/<kernelName>_in_<i>.npy / _out_<i>.npy.
extern "C" void hostLaunchSetDumpIntermediatesDir(const char *dir);

// Optional dir for per-kernel timing files; pass nullptr to disable.
// One JSON per launch is written to DIR/<kernelName>.timing.json with
// {"kernel","wall_us","block_dim","num_inputs","num_outputs","backend"}.
// Wall-clock is taken around ExecutionSession::run; on sim this is camodel
// time (not representative of NPU perf), on real NPU it includes dispatch
// overhead. Treat as a relative signal, not a precise kernel-time metric.
extern "C" void hostLaunchSetProfileDir(const char *dir);

} // namespace mlir::runtime
