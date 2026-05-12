//===- ScheduleDecision.cpp - Ascend schedule decisions ----------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Schedule/ScheduleDecision.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <utility>

namespace mlir::afir::ascend::schedule {
namespace {

void printTileShape(llvm::ArrayRef<int64_t> tileSizes,
                    llvm::raw_ostream &os) {
  os << "[";
  for (auto [index, tileSize] : llvm::enumerate(tileSizes)) {
    if (index != 0)
      os << ",";
    if (ShapedType::isDynamic(tileSize))
      os << "?";
    else
      os << tileSize;
  }
  os << "]";
}

} // namespace

ScheduleDecisionSet buildScheduleDecisionSet(
    llvm::StringRef kernelId, llvm::ArrayRef<ScheduleInstance> instances,
    const ScheduleSearchOptions &options) {
  ScheduleDecisionSet decisionSet;
  decisionSet.kernelId = kernelId.str();
  for (auto [index, instance] : llvm::enumerate(instances)) {
    ScheduleDecision decision;
    decision.decisionId =
        (llvm::Twine(kernelId) + ".decision." + llvm::Twine(index)).str();
    decision.instance = instance;
    decision.candidateGuards = instance.candidateGuards;
    decision.decisionGuards = instance.decisionGuards;
    decisionSet.decisions.push_back(std::move(decision));
  }

  if (decisionSet.decisions.empty()) {
    decisionSet.runtimeTopK = 0;
    return decisionSet;
  }

  unsigned requestedRuntimeTopK = std::max(1u, options.runtimeTopK);
  decisionSet.runtimeTopK =
      std::min<unsigned>(requestedRuntimeTopK, decisionSet.decisions.size());
  return decisionSet;
}

void printScheduleDecisionSetReport(const ScheduleDecisionSet &decisionSet,
                                    llvm::raw_ostream &os) {
  os << "ScheduleDecisionSet:\n";
  os << "  kernel = " << decisionSet.kernelId << "\n";
  os << "  decisions = " << decisionSet.decisions.size() << "\n";
  os << "  runtime_top_k = " << decisionSet.runtimeTopK << "\n";
  os << "  selected = ";
  if (decisionSet.decisions.empty())
    os << "<none>\n";
  else {
    os << decisionSet.decisions.front().decisionId << "\n";
    const ScheduleDecision &selectedDecision = decisionSet.decisions.front();
    os << "  candidate_guards = "
       << selectedDecision.candidateGuards.size() << "\n";
    os << "  decision_guards = " << selectedDecision.decisionGuards.size()
       << "\n";
    os << "  selected_tile_shape = ";
    printTileShape(selectedDecision.instance.tileShape.tileSizes, os);
    os << "\n";
  }
}

} // namespace mlir::afir::ascend::schedule
