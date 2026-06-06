//===- DebugOptions.cpp - Ascend debug options -------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Debug/DebugOptions.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace mlir::ascend::debug {
namespace {

constexpr DebugStepInfo kNormalizeSteps[] = {
    {"010-normalize-prep-out", "normalize.prep",
     "Canonicalize tensor/linalg input",
     "Run generic linalg cleanup before Ascend-specific normalization.",
     "source MLIR", "canonical linalg/tensor IR",
     "Check whether named ops were generalized and obvious CSE/canonicalize "
     "noise disappeared.",
     "Unsupported high-level op remains before Kernelize."},
    {"020-normalize-out", "normalize.output-boundary",
     "Normalize output boundary",
     "Provide the stable normalized IR consumed by Kernelize.",
     "canonical linalg/tensor IR",
     "ascend.normalized plus function symbol constraints",
     "Check function attrs, symbol constraints, and linalg.generic bodies before "
     "kernel grouping.",
     "Missing ascend.normalized marker, invalid symbol constraints, or "
     "unsupported dialect survives."},
};

constexpr DebugStepInfo kKernelizeSteps[] = {
    {"021-kernelize-structured-ops", "kernelize.structured-ops",
     "Identify kernelizable ops",
     "Find supported structured/tensor/arith ops that can participate in "
     "kernel formation.",
     "normalized tensor/linalg IR",
     "semantic participation facts for dependency analysis",
     "Check which ops are analyzed, transparent, ignored, or rejected.",
     "An expected producer is unsupported or disappeared from dependency "
     "analysis."},
    {"022-kernelize-structural-marking", "kernelize.structural-marking",
     "Mark structural relationships",
     "Mark roots, transparent view chains, branch/merge groups, and required "
     "co-location/separation constraints.",
     "dependency graph", "structural markers on participating ops",
     "Check view-like ops, branch groups, and boundary roots before role "
     "classification.",
     "A view chain or shared input creates a wrong boundary or missing edge."},
    {"023-kernelize-role-classification", "kernelize.role-classification",
     "Classify op roles",
     "Assign Vector, Cube, Reduction, Memory, and Primary roles that drive "
     "kernel candidate construction.",
     "structurally marked dependency graph", "ascend.op_roles and primary role markers",
     "Check whether the intended compute root became Primary and got the "
     "expected role.",
     "Role mismatch prevents fusion or sends the op to the wrong template "
     "family."},
    {"024-kernelize-final-patterns", "kernelize.final-patterns",
     "Build final kernel patterns",
     "Merge candidates, choose kernel partitions, and attach kernel ids.",
     "role-classified candidate graph",
     "ascend.kernel, ascend.primary, template family metadata, kernel graph edges",
     "Check kernel ids, primary ops, template families, and cross-kernel edges.",
     "Unexpected split/merge, missing primary op, or incorrect kernel graph "
     "edge."},
};

constexpr DebugStepInfo kScheduleSteps[] = {
    {"031-schedule-cleared", "schedule.clear",
     "Clear stale schedule metadata",
     "Remove previous schedule attrs so the current run cannot reuse stale "
     "decision data.",
     "kernelized IR with optional old schedule attrs",
     "kernelized IR without owned schedule attrs",
     "Check that old decision_id, tile_params, tail_plan, and target policy "
     "attrs are gone.",
     "Old schedule attrs survive and make later steps look successful for the "
     "wrong reason."},
    {"032-schedule-decisions", "schedule.choose-plan",
     "Choose tile and tail plan",
     "Build schedule problems, match templates, search candidates, and select "
     "the runtime contract.",
     "kernel patterns and target model", "schedule candidates, selected decision, tile_params, tail_plan",
     "Check ScheduleDecisionSet, tile_params, tail_policies, and tuning cache "
     "keys.",
     "No template, empty candidates, guard budget pruning, or missing target "
     "model data."},
    {"033-schedule-final", "schedule.attach-contract",
     "Attach schedule contract",
     "Attach the chosen schedule decision as the downstream symbolic runtime "
     "contract.",
     "selected schedule decision",
     "decision_id, tile_params, tail_plan, tail_policies, target_tile_policy, schedule_contract, structured_lowering",
     "Primary ops and func attrs should expose the same decision_id, "
     "tile_params, tail_policies, target_tile_policy, schedule_contract, and "
     "structured_lowering.",
     "Kernel metadata mismatch, missing tile_params, missing schedule_contract, "
     "missing structured_lowering, or incomplete tail plan."},
};

constexpr DebugStepInfo kRealizeSteps[] = {
    {"041-realize-planned", "realize.plan-memory",
     "Build memory realization plan",
     "Plan buffer values, placement, movement, workspace, and realization "
     "without mutating tensor semantics.",
     "scheduled tensor IR", "placement plan, movement plan, static memory plan",
     "Check planned GM/on-chip places, movement demands, workspace slots, and "
     "deferred decisions.",
     "Missing schedule contract, impossible placement, or unsupported movement "
     "path."},
    {"042-realize-bufferized", "realize.bufferize",
     "Bufferize tensor IR",
     "Convert tensor values to memrefs while preserving producer/consumer and "
     "view-chain relationships.",
     "planned tensor IR", "memref IR with explicit buffers and view chains",
     "Check new allocs, subviews, copies, and whether linalg inputs/outs still "
     "match the planned values.",
     "Bufferization failure, lost view-chain relation, or unexpected temporary "
     "buffer."},
    {"043-realize-memory-space-annotated", "realize.annotate-memory-space",
     "Annotate memory spaces",
     "Materialize the memory plan by assigning GM/VECIN/VECCALC/VECOUT spaces "
     "and required copies/views.",
     "bufferized memref IR", "memory_space attrs, workspace layout, materialized movement",
     "Check alloc/copy nodes, memory_space attrs, workspace slots, and deferred "
     "movement counts.",
     "Half-applied bridge, missing output copy, invalid dynamic view rewrite, "
     "or workspace reuse error."},
};

void printDebugStep(raw_ostream &os, const DebugStepInfo &step) {
  os << "  step = \"" << step.id << "\"\n";
  os << "  title = \"" << step.title << "\"\n";
  os << "  purpose = \"" << step.purpose << "\"\n";
  os << "  inputs = \"" << step.inputs << "\"\n";
  os << "  outputs = \"" << step.outputs << "\"\n";
  os << "  inspect_hint = \"" << step.inspectHint << "\"\n";
  os << "  common_failures = \"" << step.commonFailures << "\"\n";
}

void printDebugStepComment(raw_ostream &os, const DebugStepInfo &step) {
  os << "// Ascend DebugStep: " << step.id << "\n";
  os << "// Title: " << step.title << "\n";
  os << "// Purpose: " << step.purpose << "\n";
  os << "// Inputs: " << step.inputs << "\n";
  os << "// Outputs: " << step.outputs << "\n";
  os << "// Inspect: " << step.inspectHint << "\n";
  os << "// Common failures: " << step.commonFailures << "\n";
}

} // namespace

DebugStage parseDebugStage(StringRef value) {
  StringRef trimmed = value.trim();
  if (trimmed.equals_insensitive("normalize"))
    return DebugStage::Normalize;
  if (trimmed.equals_insensitive("kernelize"))
    return DebugStage::Kernelize;
  if (trimmed.equals_insensitive("schedule"))
    return DebugStage::Schedule;
  if (trimmed.equals_insensitive("realize"))
    return DebugStage::Realize;
  if (trimmed.equals_insensitive("all"))
    return DebugStage::All;
  return DebugStage::None;
}

bool shouldDump(DebugOptions options, DebugStage stage) {
  if (!options.dumpReport)
    return false;
  return options.stage == DebugStage::All || options.stage == stage;
}

bool shouldDumpCheckpoint(const DebugOptions &options, DebugStage stage) {
  if (options.checkpointDumpDir.empty())
    return false;
  return options.stage == DebugStage::All || options.stage == stage;
}

void emitStageHeader(raw_ostream &os, DebugStage stage, StringRef passName) {
  StringRef stageName = "none";
  switch (stage) {
  case DebugStage::None:
    stageName = "none";
    break;
  case DebugStage::Normalize:
    stageName = "normalize";
    break;
  case DebugStage::Kernelize:
    stageName = "kernelize";
    break;
  case DebugStage::Schedule:
    stageName = "schedule";
    break;
  case DebugStage::Realize:
    stageName = "realize";
    break;
  case DebugStage::All:
    stageName = "all";
    break;
  }
  os << "Ascend " << stageName << " report";
  if (!passName.empty())
    os << " (" << passName << ")";
  os << "\n";
  emitDebugStepCatalog(os, stage);
}

ArrayRef<DebugStepInfo> getDebugStepCatalog(DebugStage stage) {
  switch (stage) {
  case DebugStage::Normalize:
    return kNormalizeSteps;
  case DebugStage::Kernelize:
    return kKernelizeSteps;
  case DebugStage::Schedule:
    return kScheduleSteps;
  case DebugStage::Realize:
    return kRealizeSteps;
  case DebugStage::None:
  case DebugStage::All:
    return {};
  }
  return {};
}

const DebugStepInfo *lookupDebugStepInfo(StringRef checkpoint) {
  for (DebugStage stage : {DebugStage::Normalize, DebugStage::Kernelize,
                           DebugStage::Schedule, DebugStage::Realize}) {
    for (const DebugStepInfo &step : getDebugStepCatalog(stage)) {
      if (step.checkpoint == checkpoint)
        return &step;
    }
  }
  return nullptr;
}

void emitDebugStepCatalog(raw_ostream &os, DebugStage stage) {
  ArrayRef<DebugStepInfo> steps = getDebugStepCatalog(stage);
  if (steps.empty())
    return;
  os << "DebugStepCatalog:\n";
  for (const DebugStepInfo &step : steps)
    printDebugStep(os, step);
}

LogicalResult dumpCheckpoint(ModuleOp module, const DebugOptions &options,
                             DebugStage stage, StringRef basename) {
  if (!shouldDumpCheckpoint(options, stage))
    return success();
  if (basename.empty()) {
    module.emitError() << "empty Ascend debug checkpoint name";
    return failure();
  }

  std::error_code ec =
      llvm::sys::fs::create_directories(options.checkpointDumpDir);
  if (ec) {
    module.emitError() << "failed to create Ascend debug checkpoint directory '"
                       << options.checkpointDumpDir << "': " << ec.message();
    return failure();
  }

  llvm::SmallString<256> path(options.checkpointDumpDir);
  llvm::sys::path::append(path, (basename + ".mlir").str());

  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    module.emitError() << "failed to open Ascend debug checkpoint '" << path
                       << "': " << ec.message();
    return failure();
  }

  if (const DebugStepInfo *step = lookupDebugStepInfo(basename))
    printDebugStepComment(os, *step);
  module.print(os);
  os << "\n";
  return success();
}

} // namespace mlir::ascend::debug
