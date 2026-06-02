//===- HandwrittenContractRegistry.cpp - Handwritten pattern contracts ----===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/Pattern/HandwrittenContractRegistry.h"

#include "Conversion/Ascend/Kernelize/KernelizeTypes.h"
#include "Conversion/Ascend/Common/Attributes.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ManagedStatic.h"

#include <mutex>

namespace mlir::ascend::kernelize {
namespace {

struct HandwrittenContractRegistrySingleton {
  std::mutex mu;
  llvm::StringMap<HandwrittenContract> contracts;
  bool builtinsRegistered = false;
};

llvm::ManagedStatic<HandwrittenContractRegistrySingleton> gContractRegistry;

HandwrittenContractRegistrySingleton &contractRegistry() {
  return *gContractRegistry;
}

} // namespace

void registerHandwrittenContract(llvm::StringRef kind,
                                 HandwrittenContract contract) {
  HandwrittenContractRegistrySingleton &reg = contractRegistry();
  std::lock_guard<std::mutex> lock(reg.mu);
  reg.contracts.try_emplace(kind, std::move(contract));
}

void registerBuiltinHandwrittenContracts() {
  HandwrittenContractRegistrySingleton &reg = contractRegistry();
  {
    std::lock_guard<std::mutex> lock(reg.mu);
    if (reg.builtinsRegistered)
      return;
    reg.builtinsRegistered = true;
  }

  HandwrittenContract attnContract;
  attnContract.minCubeCount = 2;
  attnContract.maxCubeCount = 2;
  attnContract.requiresReduction = true;
  attnContract.requiresVectorInjective = true;
  attnContract.useAxisCarrierOnly = true;
  attnContract.primarySelectionRole = "Cube";
  attnContract.structureConstraints = {"handwritten_group", "attention_sdpa_chain"};

  HandwrittenContract::TemplateSpec &tmpl = attnContract.scheduleTemplate;
  tmpl.kindId = kKernelizeHandwrittenKindAttentionSdpa.str();
  tmpl.tilingLayout = kScheduleTemplateGroupedTilePerBlock.str();
  tmpl.tags = {kKernelizeHandwrittenKindAttentionSdpa.str(),
               kOpRoleCube.str(), kOpRoleReduction.str(),
               kOpRoleVector.str()};
  tmpl.minRank = 2;
  tmpl.maxRank = 4;
  tmpl.priority = 2;

  registerHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa,
                              std::move(attnContract));
}

const HandwrittenContract *lookupHandwrittenContract(llvm::StringRef kind) {
  if (kind.empty())
    return nullptr;
  registerBuiltinHandwrittenContracts();
  HandwrittenContractRegistrySingleton &reg = contractRegistry();
  std::lock_guard<std::mutex> lock(reg.mu);
  auto it = reg.contracts.find(kind);
  return it != reg.contracts.end() ? &it->second : nullptr;
}

} // namespace mlir::ascend::kernelize
