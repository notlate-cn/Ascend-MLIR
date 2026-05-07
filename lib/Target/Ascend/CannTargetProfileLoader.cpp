//===- CannTargetProfileLoader.cpp - CANN target profile loader -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/CannTargetProfileLoader.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <utility>

using namespace mlir;

namespace mlir::ascend {
namespace {

using SectionMap = llvm::StringMap<llvm::StringMap<std::string>>;

StringRef trim(StringRef value) { return value.trim(" \t\r\n"); }

std::string getValue(const SectionMap &sections, StringRef section,
                     StringRef key) {
  auto sectionIt = sections.find(section);
  if (sectionIt == sections.end())
    return "";
  auto keyIt = sectionIt->second.find(key);
  if (keyIt == sectionIt->second.end())
    return "";
  return keyIt->second;
}

int64_t parseInt64(StringRef value) {
  int64_t parsed = 0;
  if (trim(value).getAsInteger(0, parsed))
    return 0;
  return parsed;
}

bool parseBool(StringRef value) {
  return llvm::StringSwitch<bool>(trim(value).lower())
      .Cases("1", "true", "yes", "on", true)
      .Default(false);
}

SmallVector<std::string> parseDtypes(StringRef value) {
  SmallVector<std::string> dtypes;
  SmallVector<StringRef> tokens;
  value.split(tokens, ',');
  if (tokens.size() == 1) {
    tokens.clear();
    value.split(tokens, ' ');
  }
  for (StringRef token : tokens) {
    token = trim(token);
    if (!token.empty())
      dtypes.push_back(token.str());
  }
  return dtypes;
}

void parseIni(StringRef content, SectionMap &sections) {
  std::string currentSection;
  SmallVector<StringRef> lines;
  content.split(lines, '\n');
  for (StringRef line : lines) {
    line = trim(line);
    if (line.empty() || line.starts_with("#") || line.starts_with(";"))
      continue;

    if (line.starts_with("[") && line.ends_with("]")) {
      currentSection = trim(line.drop_front().drop_back()).str();
      continue;
    }

    std::pair<StringRef, StringRef> entry = line.split('=');
    if (entry.second.empty())
      continue;
    sections[currentSection][trim(entry.first)] = trim(entry.second).str();
  }
}

void setCapacity(TargetProfile &profile, MemoryPlace place, int64_t bytes) {
  if (bytes > 0)
    profile.capacityBytes[place] = bytes;
}

FailureOr<TargetProfile> loadImpl(StringRef cannRoot, StringRef socVersion,
                                  raw_ostream &os) {
  if (cannRoot.empty()) {
    os << "failed to load TargetProfile: cann-root is empty\n";
    return failure();
  }

  SmallString<256> iniPath(cannRoot);
  SmallString<64> iniFile(socVersion);
  iniFile += ".ini";
  llvm::sys::path::append(iniPath, "aarch64-linux", "data", "platform_config",
                          iniFile);

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(iniPath);
  if (!buffer) {
    os << "failed to load TargetProfile: cannot open " << iniPath << ": "
       << buffer.getError().message() << "\n";
    return failure();
  }

  SectionMap sections;
  parseIni(buffer.get()->getBuffer(), sections);

  TargetProfile profile;
  profile.identity.socVersion =
      getValue(sections, "version", "SoC_version").empty()
          ? socVersion.str()
          : getValue(sections, "version", "SoC_version");
  profile.identity.shortSocVersion =
      getValue(sections, "version", "Short_SoC_version");
  profile.identity.npuArch = getValue(sections, "version", "NpuArch");

  profile.hardware.aiCoreCount =
      parseInt64(getValue(sections, "SoCInfo", "ai_core_cnt"));
  profile.hardware.cubeCoreCount =
      parseInt64(getValue(sections, "SoCInfo", "cube_core_cnt"));
  profile.hardware.vectorCoreCount =
      parseInt64(getValue(sections, "SoCInfo", "vector_core_cnt"));
  profile.hardware.supportBF16 =
      parseBool(getValue(sections, "SoCInfo", "support_bf16"));
  profile.hardware.supportFixpipe =
      parseBool(getValue(sections, "AICoreSpec", "support_fixpipe"));
  profile.hardware.l1SizeBytes =
      parseInt64(getValue(sections, "AICoreSpec", "l1_size"));
  profile.hardware.ubSizeBytes =
      parseInt64(getValue(sections, "AICoreSpec", "ub_size"));

  setCapacity(profile, MemoryPlace::GM,
              parseInt64(getValue(sections, "SoCInfo", "memory_size")));
  setCapacity(profile, MemoryPlace::L2,
              parseInt64(getValue(sections, "SoCInfo", "l2_size")));
  setCapacity(profile, MemoryPlace::L1, profile.hardware.l1SizeBytes);
  setCapacity(profile, MemoryPlace::L0A,
              parseInt64(getValue(sections, "AICoreSpec", "l0_a_size")));
  setCapacity(profile, MemoryPlace::L0B,
              parseInt64(getValue(sections, "AICoreSpec", "l0_b_size")));
  setCapacity(profile, MemoryPlace::L0C,
              parseInt64(getValue(sections, "AICoreSpec", "l0_c_size")));
  setCapacity(profile, MemoryPlace::UB, profile.hardware.ubSizeBytes);

  profile.movementPaths.push_back({MemoryPlace::GM, MemoryPlace::L1});
  profile.movementPaths.push_back({MemoryPlace::L1, MemoryPlace::L0A});
  profile.movementPaths.push_back({MemoryPlace::L1, MemoryPlace::L0B});
  profile.movementPaths.push_back({MemoryPlace::L0C, MemoryPlace::UB});
  profile.movementPaths.push_back({MemoryPlace::UB, MemoryPlace::GM});

  llvm::StringSet<> seenIntrinsics;
  auto dtypeSectionIt = sections.find("AICoreintrinsicDtypeMap");
  if (dtypeSectionIt != sections.end()) {
    for (const auto &entry : dtypeSectionIt->second) {
      if (!entry.first().starts_with("Intrinsic_"))
        continue;
      if (!seenIntrinsics.insert(entry.first()).second)
        continue;
      TargetIntrinsicInfo intrinsic;
      intrinsic.name = entry.first().str();
      intrinsic.dtypes = parseDtypes(entry.second);
      profile.intrinsics.push_back(std::move(intrinsic));
    }
  }

  auto ratesSectionIt = sections.find("AICoreMemoryRates");
  if (ratesSectionIt != sections.end()) {
    for (const auto &entry : ratesSectionIt->second) {
      if (!entry.first().starts_with("Intrinsic_"))
        continue;
      if (!seenIntrinsics.insert(entry.first()).second)
        continue;
      TargetIntrinsicInfo intrinsic;
      intrinsic.name = entry.first().str();
      intrinsic.dtypes = parseDtypes(entry.second);
      profile.intrinsics.push_back(std::move(intrinsic));
    }
  }

  if (failed(verifyTargetProfile(profile, os)))
    return failure();
  return profile;
}

} // namespace

FailureOr<TargetProfile>
CannTargetProfileLoader::load(StringRef cannRoot, StringRef socVersion) {
  return loadImpl(cannRoot, socVersion, llvm::errs());
}

} // namespace mlir::ascend
