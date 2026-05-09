//===- TemplateRegistry.cpp - Ascend schedule templates ----------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Schedule/TemplateRegistry.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::afir::ascend::schedule {
namespace {

ArrayRef<ScheduleTemplate> getRegistryTemplates() {
  static const SmallVector<ScheduleTemplate> templates = {
      {"vector_static_1d", "single_tile_per_block", {"vector"}, 1, 1, 0},
      {"vector_static_2d", "single_tile_per_block", {"vector"}, 2, 2, 1},
      {"reduction_static", "single_tile_per_block", {"reduction"}, 0, 8, 2},
      {"cube_static_matmul", "single_tile_per_block", {"cube"}, 2, 2, 3},
  };
  return templates;
}

bool hasTag(ArrayRef<std::string> tags, StringRef tag) {
  return llvm::any_of(tags, [&](const std::string &candidate) {
    return StringRef(candidate) == tag;
  });
}

bool tagsIntersect(ArrayRef<std::string> lhs, ArrayRef<std::string> rhs) {
  return llvm::any_of(lhs, [&](const std::string &tag) {
    return hasTag(rhs, tag);
  });
}

void sortTemplates(SmallVectorImpl<ScheduleTemplate> &templates) {
  llvm::sort(templates, [](const ScheduleTemplate &lhs,
                           const ScheduleTemplate &rhs) {
    if (lhs.priority != rhs.priority)
      return lhs.priority < rhs.priority;
    if (lhs.family != rhs.family)
      return lhs.family < rhs.family;
    return lhs.name < rhs.name;
  });
}

} // namespace

SmallVector<ScheduleTemplate>
matchScheduleTemplates(const ScheduleProblem &problem) {
  SmallVector<ScheduleTemplate> matches;
  for (const ScheduleTemplate &scheduleTemplate : getRegistryTemplates()) {
    if (problem.resultRank < scheduleTemplate.minRank ||
        problem.resultRank > scheduleTemplate.maxRank)
      continue;
    if (!tagsIntersect(scheduleTemplate.tags, problem.templateTags))
      continue;
    matches.push_back(scheduleTemplate);
  }

  SmallVector<ScheduleTemplate> exactRoleMatches;
  StringRef roleTag = stringifyOpRole(problem.dominantRole);
  for (const ScheduleTemplate &scheduleTemplate : matches) {
    if (hasTag(scheduleTemplate.tags, roleTag))
      exactRoleMatches.push_back(scheduleTemplate);
  }
  if (!exactRoleMatches.empty())
    matches = std::move(exactRoleMatches);

  sortTemplates(matches);
  return matches;
}

void printTemplateRegistryReport(StringRef kernelId,
                                 ArrayRef<ScheduleTemplate> matches,
                                 llvm::raw_ostream &os) {
  os << "TemplateRegistry:\n";
  os << "  kernel = " << kernelId << "\n";
  os << "  matches = " << matches.size() << "\n";
  for (const ScheduleTemplate &scheduleTemplate : matches)
    os << "  template = " << scheduleTemplate.family << "/"
       << scheduleTemplate.name << "\n";
}

} // namespace mlir::afir::ascend::schedule
