//===- TemplateRegistry.cpp - Ascend schedule templates ----------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "TemplateRegistry.h"

#include "Conversion/Ascend/Kernelize/Pattern/HandwrittenContractRegistry.h"
#include "ScheduleTemplateImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <mutex>
#include <utility>

namespace mlir::ascend::schedule {
namespace {

struct TemplateRegistrySingleton {
  std::mutex mu;
  SmallVector<std::unique_ptr<ScheduleTemplateImplementation>> templates;
  bool builtinsRegistered = false;
};

llvm::ManagedStatic<TemplateRegistrySingleton> gRegistry;

TemplateRegistrySingleton &registry() { return *gRegistry; }

void appendTemplateImplementation(
    TemplateRegistrySingleton &reg,
    std::unique_ptr<ScheduleTemplateImplementation> implementation) {
  reg.templates.push_back(std::move(implementation));
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

void sortTemplateImplementations(
    SmallVectorImpl<const ScheduleTemplateImplementation *> &templates) {
  llvm::sort(templates, [](const ScheduleTemplateImplementation *lhs,
                           const ScheduleTemplateImplementation *rhs) {
    const ScheduleTemplate &lhsMetadata = lhs->metadata();
    const ScheduleTemplate &rhsMetadata = rhs->metadata();
    if (lhsMetadata.priority != rhsMetadata.priority)
      return lhsMetadata.priority < rhsMetadata.priority;
    if (lhsMetadata.family != rhsMetadata.family)
      return lhsMetadata.family < rhsMetadata.family;
    return lhsMetadata.name < rhsMetadata.name;
  });
}

} // namespace

void registerBuiltinTemplates() {
  TemplateRegistrySingleton &reg = registry();
  std::lock_guard<std::mutex> lock(reg.mu);
  if (reg.builtinsRegistered)
    return;
  reg.builtinsRegistered = true;

  appendTemplateImplementation(
      reg, createSingleTilePerBlockTemplateImplementation(
               kScheduleFamilyVectorGeneric, kOpRoleVector, 1, 8, 0));
  appendTemplateImplementation(
      reg, createSingleTilePerBlockTemplateImplementation(
               kScheduleFamilyReductionStatic, kOpRoleReduction, 0, 8, 2));
  {
    using namespace ::mlir::ascend::kernelize;
    registerBuiltinHandwrittenContracts();
    for (llvm::StringRef kind : {kKernelizeHandwrittenKindAttentionSdpa}) {
      const HandwrittenContract *contract = lookupHandwrittenContract(kind);
      if (!contract)
        continue;
      const HandwrittenContract::TemplateSpec &spec = contract->scheduleTemplate;
      ScheduleTemplate tmpl;
      tmpl.family = spec.kindId;
      tmpl.name = spec.tilingLayout;
      for (const std::string &tag : spec.tags)
        tmpl.tags.push_back(tag);
      tmpl.minRank = spec.minRank;
      tmpl.maxRank = spec.maxRank;
      tmpl.priority = spec.priority;
      appendTemplateImplementation(
          reg, createGroupedTilePerBlockTemplateImplementation(std::move(tmpl)));
    }
  }
  appendTemplateImplementation(
      reg, createSingleTilePerBlockTemplateImplementation(
               kScheduleFamilyCubeStaticMatmul, kOpRoleCube, 2, 3, 3));
  appendTemplateImplementation(
      reg, createSingleTilePerBlockTemplateImplementation(
               kScheduleFamilyMemoryCopy, kOpRoleMemory, 0, 8, 4));
}

void registerTemplate(ScheduleTemplate tmpl) {
  registerTemplateImplementation(createRoleDrivenTemplateImplementation(
      std::move(tmpl), "role-driven schedule template"));
}

void registerTemplateImplementation(
    std::unique_ptr<ScheduleTemplateImplementation> implementation) {
  TemplateRegistrySingleton &reg = registry();
  std::lock_guard<std::mutex> lock(reg.mu);
  appendTemplateImplementation(reg, std::move(implementation));
}

SmallVector<const ScheduleTemplateImplementation *>
matchScheduleTemplateImplementations(const ScheduleProblem &problem) {
  registerBuiltinTemplates();

  TemplateRegistrySingleton &reg = registry();
  std::lock_guard<std::mutex> lock(reg.mu);

  SmallVector<const ScheduleTemplateImplementation *> matches;
  for (const std::unique_ptr<ScheduleTemplateImplementation> &implementation :
       reg.templates) {
    const ScheduleTemplate &scheduleTemplate = implementation->metadata();
    if (problem.resultRank < scheduleTemplate.minRank ||
        problem.resultRank > scheduleTemplate.maxRank)
      continue;
    if (!hasRequiredContractTags(scheduleTemplate, problem.templateTags))
      continue;
    if (!tagsIntersect(scheduleTemplate.tags, problem.templateTags))
      continue;
    matches.push_back(implementation.get());
  }

  SmallVector<const ScheduleTemplateImplementation *> exactRoleMatches;
  StringRef roleTag = stringifyOpRole(problem.dominantRole);
  for (const ScheduleTemplateImplementation *implementation : matches) {
    const ScheduleTemplate &scheduleTemplate = implementation->metadata();
    if (hasTag(scheduleTemplate.tags, roleTag))
      exactRoleMatches.push_back(implementation);
  }
  if (!exactRoleMatches.empty())
    matches = std::move(exactRoleMatches);

  sortTemplateImplementations(matches);
  return matches;
}

SmallVector<ScheduleTemplate>
matchScheduleTemplates(const ScheduleProblem &problem) {
  SmallVector<const ScheduleTemplateImplementation *> implementations =
      matchScheduleTemplateImplementations(problem);
  SmallVector<ScheduleTemplate> matches;
  matches.reserve(implementations.size());
  for (const ScheduleTemplateImplementation *implementation : implementations)
    matches.push_back(implementation->metadata());
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

void printTemplateRegistryReport(
    StringRef kernelId,
    ArrayRef<const ScheduleTemplateImplementation *> matches,
    llvm::raw_ostream &os) {
  os << "TemplateRegistry:\n";
  os << "  kernel = " << kernelId << "\n";
  os << "  matches = " << matches.size() << "\n";
  for (const ScheduleTemplateImplementation *implementation : matches) {
    const ScheduleTemplate &scheduleTemplate = implementation->metadata();
    os << "  template = " << scheduleTemplate.family << "/"
       << scheduleTemplate.name << "\n";
  }
}

} // namespace mlir::ascend::schedule
