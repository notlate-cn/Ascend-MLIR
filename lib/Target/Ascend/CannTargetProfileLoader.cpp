//===- CannTargetProfileLoader.cpp - CANN target profile loader -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/CannTargetProfileLoader.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <optional>
#include <tuple>
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

std::optional<TargetIntrinsicInfo> parseIntrinsicEntry(StringRef key,
                                                       StringRef value) {
  TargetIntrinsicInfo intrinsic;
  key = trim(key);
  value = trim(value);

  if (key.starts_with("Intrinsic_")) {
    intrinsic.name = key.str();
    intrinsic.dtypes = parseDtypes(value);
    return intrinsic;
  }

  auto [name, dtypeList] = value.split('|');
  name = trim(name);
  if (!name.starts_with("Intrinsic_"))
    return std::nullopt;

  intrinsic.name = name.str();
  intrinsic.dtypes = parseDtypes(dtypeList);
  return intrinsic;
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

void appendMemoryRate(TargetProfile &profile, StringRef section,
                      StringRef name, StringRef value) {
  name = trim(name);
  if (name.empty() || name.starts_with("Intrinsic_"))
    return;

  int64_t rate = parseInt64(value);
  if (rate <= 0)
    return;
  profile.memoryRates.push_back({section.str(), name.str(), rate});
}

void parseMemoryRates(TargetProfile &profile, const SectionMap &sections,
                      StringRef section) {
  auto sectionIt = sections.find(section);
  if (sectionIt == sections.end())
    return;
  for (const auto &entry : sectionIt->second)
    appendMemoryRate(profile, section, entry.first(), entry.second);
}

void appendDType(TargetIntrinsicInfo &intrinsic, StringRef dtype) {
  if (!llvm::is_contained(intrinsic.dtypes, dtype))
    intrinsic.dtypes.push_back(dtype.str());
}

void appendUnit(TargetIntrinsicInfo &intrinsic, ExecutionUnit unit) {
  if (!llvm::is_contained(intrinsic.units, unit))
    intrinsic.units.push_back(unit);
}

bool isFixPipePathIntrinsic(StringRef name) {
  return name.starts_with("Intrinsic_fix_pipe_l");
}

void appendUnitsByName(TargetIntrinsicInfo &intrinsic) {
  StringRef name(intrinsic.name);
  if (name == "Intrinsic_mmad")
    appendUnit(intrinsic, ExecutionUnit::Cube);
  else if (name.starts_with("Intrinsic_v"))
    appendUnit(intrinsic, ExecutionUnit::Vector);
  else if (name.starts_with("Intrinsic_data_move_") ||
           isFixPipePathIntrinsic(name))
    appendUnit(intrinsic, ExecutionUnit::DMA);
}

void mergeIntrinsic(llvm::StringMap<TargetIntrinsicInfo> &intrinsics,
                    TargetIntrinsicInfo intrinsic) {
  if (intrinsic.name.empty())
    return;

  TargetIntrinsicInfo &merged = intrinsics[intrinsic.name];
  merged.name = intrinsic.name;
  for (StringRef dtype : intrinsic.dtypes)
    appendDType(merged, dtype);
  for (ExecutionUnit unit : intrinsic.units)
    appendUnit(merged, unit);
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
  setCapacity(profile, MemoryPlace::A1, profile.hardware.l1SizeBytes);
  setCapacity(profile, MemoryPlace::B1, profile.hardware.l1SizeBytes);
  setCapacity(profile, MemoryPlace::A2,
              parseInt64(getValue(sections, "AICoreSpec", "l0_a_size")));
  setCapacity(profile, MemoryPlace::B2,
              parseInt64(getValue(sections, "AICoreSpec", "l0_b_size")));
  setCapacity(profile, MemoryPlace::CO1,
              parseInt64(getValue(sections, "AICoreSpec", "l0_c_size")));
  setCapacity(profile, MemoryPlace::VECIN, profile.hardware.ubSizeBytes);
  setCapacity(profile, MemoryPlace::VECOUT, profile.hardware.ubSizeBytes);
  setCapacity(profile, MemoryPlace::VECCALC, profile.hardware.ubSizeBytes);

  parseMemoryRates(profile, sections, "AICoreMemoryRates");
  parseMemoryRates(profile, sections, "VectorCoreMemoryRates");
  llvm::sort(profile.memoryRates, [](const TargetMemoryRateInfo &lhs,
                                     const TargetMemoryRateInfo &rhs) {
    return std::tie(lhs.section, lhs.name) < std::tie(rhs.section, rhs.name);
  });

  struct IntrinsicSection {
    StringRef name;
    std::optional<ExecutionUnit> unit;
    bool inferUnitByName = false;
  };
  constexpr IntrinsicSection intrinsicSections[] = {
      {"AICoreintrinsicDtypeMap", std::nullopt, true},
      {"CUBECoreintrinsicDtypeMap", ExecutionUnit::Cube, false},
      {"VectorCoreintrinsicDtypeMap", ExecutionUnit::Vector, false},
  };

  llvm::StringMap<TargetIntrinsicInfo> intrinsicMap;
  for (const IntrinsicSection &section : intrinsicSections) {
    auto dtypeSectionIt = sections.find(section.name);
    if (dtypeSectionIt == sections.end())
      continue;

    for (const auto &entry : dtypeSectionIt->second) {
      std::optional<TargetIntrinsicInfo> parsed =
          parseIntrinsicEntry(entry.first(), entry.second);
      if (parsed) {
        if (section.unit)
          appendUnit(*parsed, *section.unit);
        if (section.inferUnitByName)
          appendUnitsByName(*parsed);
        mergeIntrinsic(intrinsicMap, std::move(*parsed));
      }
    }
  }

  auto ratesSectionIt = sections.find("AICoreMemoryRates");
  if (ratesSectionIt != sections.end()) {
    for (const auto &entry : ratesSectionIt->second) {
      if (!entry.first().starts_with("Intrinsic_"))
        continue;
      TargetIntrinsicInfo intrinsic;
      intrinsic.name = entry.first().str();
      appendUnitsByName(intrinsic);
      mergeIntrinsic(intrinsicMap, std::move(intrinsic));
    }
  }

  for (auto &entry : intrinsicMap) {
    llvm::sort(entry.second.dtypes);
    entry.second.dtypes.erase(
        std::unique(entry.second.dtypes.begin(), entry.second.dtypes.end()),
        entry.second.dtypes.end());
    profile.intrinsics.push_back(std::move(entry.second));
  }

  llvm::sort(profile.intrinsics, [](const TargetIntrinsicInfo &lhs,
                                    const TargetIntrinsicInfo &rhs) {
    return lhs.name < rhs.name;
  });

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
