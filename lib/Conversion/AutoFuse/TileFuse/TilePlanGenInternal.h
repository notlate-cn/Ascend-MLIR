#ifndef CONVERSION_AUTOFUSE_TILEFUSE_TILEPLANGENINTERNAL_H
#define CONVERSION_AUTOFUSE_TILEFUSE_TILEPLANGENINTERNAL_H

#include "Conversion/AutoFuse/GroupInfo.h"
#include "Conversion/AutoFuse/TilePlan.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

// Internal contract for the TilePlanGen god-file split: symbols extracted out of
// TilePlanGen.cpp's anonymous namespace that are still called across the new
// TUs (TilePlanCost.cpp defines them; TilePlanGen.cpp's plan-build/emit roots
// call them).  Not a public API — kept under lib/, not include/.

namespace mlir::afir {

// §3.6 cost-model SoC capacity (UB bytes / #AI-cores).
struct SocConstants {
  int64_t ubBytes;
  int64_t numAICores;
};

// The block-dispatch axis pick.
struct BlockPick {
  int  axis = -1;
  bool degradeToRowLoop = false;
};

// Score sentinel: a draft scoring >= kInfeasible is rejected by the picker.
extern const double kInfeasible;

SocConstants getSocConstants(llvm::StringRef socName);

unsigned operandElemBytes(const auto_fuse::CollapsedGroupInfo &info);

llvm::DenseSet<int>
computeVectorizedDims(const auto_fuse::CollapsedGroupInfo &info,
                      auto_fuse::TilePlan *plan = nullptr);

BlockPick pickBlockAxis(const auto_fuse::AxisGrouping &g,
                        const llvm::DenseSet<int> &vecDims);

llvm::SmallVector<auto_fuse::TilePlanDraft>
enumerateTilingCases(const auto_fuse::AxisGrouping &g,
                     const auto_fuse::CollapsedGroupInfo &info,
                     bool enableReductionSplit, unsigned elemBytes);

double costEstimate(const auto_fuse::AxisGrouping &g,
                    const auto_fuse::CollapsedGroupInfo &info,
                    const llvm::DenseSet<int> &vecDims,
                    const auto_fuse::TilePlanDraft &draft, unsigned elemBytes,
                    SocConstants soc, bool relaxNonBlockUbY = false);

// Classify the trailing bias/activation epilogue chain after a linalg.matmul.
// Returns {hasBias, epilogueKindName}.  Defined in TilePlanCubeAnalysis.cpp.
std::pair<bool, llvm::StringRef>
classifyCubeEpilogueChain(func::FuncOp func, auto_fuse::CubeKind cubeKind);

} // namespace mlir::afir

#endif // CONVERSION_AUTOFUSE_TILEFUSE_TILEPLANGENINTERNAL_H
