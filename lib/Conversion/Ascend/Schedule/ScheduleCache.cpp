//===- ScheduleCache.cpp - Ascend schedule cache model ----------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ScheduleCache.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::afir::ascend::schedule {
namespace {

std::string serializeDims(ArrayRef<int64_t> dims) {
  std::string result;
  llvm::raw_string_ostream os(result);
  llvm::interleave(
      dims, os,
      [&](int64_t dim) {
        if (ShapedType::isDynamic(dim))
          os << "?";
        else
          os << dim;
      },
      "x");
  return result;
}

std::string serializeShapeBucketKey(const ShapeBucketKey &key) {
  return (llvm::Twine(key.kernelId) + "|" + key.family + "|" +
          serializeDims(key.resultShape))
      .str();
}

std::string serializeTuningResultKey(const TuningResultKey &key) {
  return (llvm::Twine(key.bucket.kernelId) + "|" + key.bucket.family + "|" +
          key.templateName + "|" + serializeDims(key.bucket.resultShape) +
          "|" + serializeDims(key.tileShape))
      .str();
}

std::string getShapeBucketSignature(const ShapeBucketKey &key) {
  return (llvm::Twine(key.family) + "|" + serializeDims(key.resultShape)).str();
}

std::string getTuningResultSignature(const TuningResultKey &key) {
  return (llvm::Twine(key.bucket.family) + "|" + key.templateName + "|" +
          serializeDims(key.bucket.resultShape) + "|" +
          serializeDims(key.tileShape))
      .str();
}

ShapeBucketKey makeShapeBucketKey(const ScheduleProblem &problem,
                                  const ScheduleInstance &instance) {
  ShapeBucketKey key;
  key.kernelId = problem.kernelId;
  key.family = instance.tmpl.family;
  key.resultShape = problem.resultShape;
  return key;
}

TuningResultKey makeTuningResultKey(ShapeBucketKey bucket,
                                    const ScheduleInstance &instance) {
  TuningResultKey key;
  key.bucket = std::move(bucket);
  key.templateName = instance.tmpl.name;
  key.tileShape = instance.tileShape.tileSizes;
  return key;
}

} // namespace

ScheduleCacheReport ScheduleCacheModel::recordScheduleDecisionSet(
    const ScheduleProblem &problem, const ScheduleDecisionSet &decisionSet) {
  for (const ScheduleDecision &decision : decisionSet.decisions) {
    ShapeBucketKey shapeKey =
        makeShapeBucketKey(problem, decision.instance);
    ++report.shapeBucketLookups;

    // kernelId is an origin/debug field. Cache identity is intentionally
    // normalized across kernel ids so identical kernels can reuse shape and
    // tuning entries within one pass run.
    if (shapeBucketSignatures.insert(getShapeBucketSignature(shapeKey)).second) {
      ++report.shapeBucketMisses;
      shapeBucketKeys.push_back(shapeKey);
    }

    TuningResultKey tuningKey =
        makeTuningResultKey(std::move(shapeKey), decision.instance);
    ++report.tuningLookups;
    if (tuningResultSignatures.insert(getTuningResultSignature(tuningKey))
            .second) {
      ++report.tuningMisses;
      tuningResultKeys.push_back(std::move(tuningKey));
    }
  }

  return report;
}

void ScheduleCacheModel::recordGuardBudgetPruned(unsigned count) {
  report.negativeCacheEntries += count;
}

void ScheduleCacheModel::recordNegativeCacheEntry() {
  ++report.negativeCacheEntries;
}

void printScheduleCacheReport(const ScheduleCacheReport &report,
                              llvm::raw_ostream &os) {
  os << "ScheduleCache:\n";
  os << "  shape_bucket_lookups = " << report.shapeBucketLookups << "\n";
  os << "  shape_bucket_misses = " << report.shapeBucketMisses << "\n";
  os << "  tuning_lookups = " << report.tuningLookups << "\n";
  os << "  tuning_misses = " << report.tuningMisses << "\n";
  os << "  negative_cache_entries = " << report.negativeCacheEntries << "\n";
}

void printScheduleCacheReport(const ScheduleCacheModel &cacheModel,
                              llvm::raw_ostream &os) {
  printScheduleCacheReport(cacheModel.getReport(), os);
  for (const ShapeBucketKey &key : cacheModel.getShapeBucketKeys())
    os << "  shape_bucket_key = " << serializeShapeBucketKey(key) << "\n";
  for (const TuningResultKey &key : cacheModel.getTuningResultKeys())
    os << "  tuning_result_key = " << serializeTuningResultKey(key) << "\n";
}

} // namespace mlir::afir::ascend::schedule
