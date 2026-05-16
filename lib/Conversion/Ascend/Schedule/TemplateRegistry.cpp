//===- TemplateRegistry.cpp - Ascend schedule templates ----------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "TemplateRegistry.h"

#include "../Kernelize/HandwrittenContractRegistry.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>

namespace mlir::afir::ascend::schedule {
namespace {

struct TemplateRegistrySingleton {
  std::mutex mu;
  SmallVector<ScheduleTemplate> templates;
  bool builtinsRegistered = false;
};

llvm::ManagedStatic<TemplateRegistrySingleton> gRegistry;

TemplateRegistrySingleton &registry() { return *gRegistry; }

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

bool isRoleTemplateTag(StringRef tag) {
  return tag == kOpRoleCube || tag == kOpRoleMemory ||
         tag == kOpRoleReduction || tag == kOpRoleVector;
}

bool hasRequiredContractTags(const ScheduleTemplate &scheduleTemplate,
                             ArrayRef<std::string> problemTags) {
  for (StringRef tag : scheduleTemplate.tags) {
    if (isRoleTemplateTag(tag))
      continue;
    if (!hasTag(problemTags, tag))
      return false;
  }
  return true;
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

void registerBuiltinTemplates() {
  TemplateRegistrySingleton &reg = registry();
  std::lock_guard<std::mutex> lock(reg.mu);
  if (reg.builtinsRegistered)
    return;
  reg.builtinsRegistered = true;

  reg.templates.push_back(
      {"vector_generic", "single_tile_per_block", {kOpRoleVector.str()}, 1, 8, 0});
  reg.templates.push_back(
      {"reduction_static", "single_tile_per_block", {kOpRoleReduction.str()}, 0, 8, 2});
  {
    using namespace ::mlir::afir::ascend::kernelize;
    registerBuiltinHandwrittenContracts();
    for (llvm::StringRef kind : {kKernelizeHandwrittenKindAttentionSdpa}) {
      const HandwrittenContract *contract = lookupHandwrittenContract(kind);
      if (!contract)
        continue;
      const HandwrittenContract::TemplateSpec &spec = contract->scheduleTemplate;
      ScheduleTemplate tmpl;
      tmpl.family = spec.layout;  // spec.layout → ScheduleTemplate::family
      tmpl.name = spec.name;      // spec.name   → ScheduleTemplate::name
      for (const std::string &tag : spec.tags)
        tmpl.tags.push_back(tag);
      tmpl.minRank = spec.minRank;
      tmpl.maxRank = spec.maxRank;
      tmpl.priority = spec.priority;
      reg.templates.push_back(std::move(tmpl));
    }
  }
  reg.templates.push_back(
      {"cube_static_matmul", "single_tile_per_block", {kOpRoleCube.str()}, 2, 3, 3});
  reg.templates.push_back(
      {"memory_copy", "single_tile_per_block", {kOpRoleMemory.str()}, 0, 8, 4});
}

void registerTemplate(ScheduleTemplate tmpl) {
  TemplateRegistrySingleton &reg = registry();
  std::lock_guard<std::mutex> lock(reg.mu);
  reg.templates.push_back(std::move(tmpl));
}

SmallVector<ScheduleTemplate>
matchScheduleTemplates(const ScheduleProblem &problem) {
  registerBuiltinTemplates();

  TemplateRegistrySingleton &reg = registry();
  std::lock_guard<std::mutex> lock(reg.mu);

  SmallVector<ScheduleTemplate> matches;
  for (const ScheduleTemplate &scheduleTemplate : reg.templates) {
    if (problem.resultRank < scheduleTemplate.minRank ||
        problem.resultRank > scheduleTemplate.maxRank)
      continue;
    if (!hasRequiredContractTags(scheduleTemplate, problem.templateTags))
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
