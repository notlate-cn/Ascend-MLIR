//===- SocSpec.h - SoC capacity constants from CANN ---------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
// Per-SoC TBuf/TQue allocator capacity constants, mirroring the
// __NPU_ARCH__-switched values in CANN
// asc/impl/basic_api/utils/kernel_utils_constants.h.  Used by
// CannTranslation to stamp ub_budget_bytes in tiling_space.json so that
// network_runner's phase-3 picker and the autotuner can prune candidate
// tilings whose UB cost exceeds budget.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_TARGET_CANNKERNEL_SOCSPEC_H
#define AFIR_TARGET_CANNKERNEL_SOCSPEC_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>

namespace mlir::afir::cannkernel {

struct SocSpec {
  uint32_t totalUbSize;       // TOTAL_UB_SIZE (raw HW)
  uint32_t totalVecLocalSize; // TOTAL_VEC_LOCAL_SIZE (user TBuf/TQue pool)
  uint32_t oneBlockSize;      // ONE_BLK_SIZE (allocator alignment)
  uint32_t numAICores;        // count of AI cores available for block dispatch
};

/// Returns the SoC capacity record for `soc`, or nullopt if unknown.
/// Source: CANN asc/impl/basic_api/utils/kernel_utils_constants.h.
inline std::optional<SocSpec> getSocSpec(llvm::StringRef soc) {
  // NPU_ARCH 2201 — Ascend910B1: 40 AI cores (8 cube + 32 vector, plan §3.6
  // "blockDim distance to #AICores" uses the vector-side count for the
  // post-collapse compute kernels we generate today).
  if (soc == "Ascend910B1")
    return SocSpec{192u * 1024, 184u * 1024, 32u, 40u};
  // NPU_ARCH 1001 / 2002 — Ascend910 / Ascend910A
  if (soc == "Ascend910" || soc == "Ascend910A")
    return SocSpec{256u * 1024, 248u * 1024, 32u, 32u};
  // NPU_ARCH 3002 — Ascend910C
  if (soc == "Ascend910C")
    return SocSpec{248u * 1024, 184u * 1024, 32u, 40u};
  // NPU_ARCH 3102
  if (soc == "Ascend910N")
    return SocSpec{256u * 1024, 184u * 1024, 32u, 40u};
  // NPU_ARCH 3510 / 5102 — Ascend910D / next gen
  if (soc == "Ascend910D")
    return SocSpec{248u * 1024, 248u * 1024, 32u, 40u};
  // NPU_ARCH 3003 / 3113 — Ascend310B (much smaller, ~8 vector cores)
  if (soc == "Ascend310B")
    return SocSpec{118u * 1024, 118u * 1024, 32u, 8u};
  return std::nullopt;
}

} // namespace mlir::afir::cannkernel

#endif // AFIR_TARGET_CANNKERNEL_SOCSPEC_H
