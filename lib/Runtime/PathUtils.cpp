#include "Runtime/PathUtils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Error.h"
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <vector>

namespace mlir::runtime {

namespace {

static std::vector<std::string> getCannArchCandidates(llvm::StringRef machine) {
  if (machine == "x86_64" || machine == "amd64")
    return {"x86_64-linux"};
  if (machine == "aarch64" || machine == "arm64")
    return {"aarch64-linux", "arm64-linux"};
  return {};
}

static std::vector<std::string> getHostCannArchCandidates() {
#if defined(__x86_64__) || defined(_M_X64)
  return {"x86_64-linux"};
#elif defined(__aarch64__) || defined(_M_ARM64)
  return {"aarch64-linux", "arm64-linux"};
#else
  return {};
#endif
}

static std::string findUnderAscendHome(llvm::StringRef ascendHome,
                                       llvm::ArrayRef<std::string> suffixes) {
  std::vector<std::string> candidates;
  candidates.reserve(suffixes.size());
  for (const std::string &suffix : suffixes)
    candidates.push_back((ascendHome + suffix).str());
  std::string resolved = findFirstExistingPath(candidates);
  if (!resolved.empty())
    return resolved;
  return candidates.empty() ? std::string() : candidates.front();
}

static std::vector<std::string>
buildArchRelativeCandidates(llvm::StringRef genericSuffix,
                            llvm::StringRef archRelativeSuffix,
                            std::optional<llvm::StringRef> machine = std::nullopt) {
  std::vector<std::string> candidates;
  const std::vector<std::string> archDirs =
      machine ? getCannArchCandidates(*machine) : getHostCannArchCandidates();
  candidates.reserve(archDirs.size() + (genericSuffix.empty() ? 0 : 1));
  for (const std::string &archDir : archDirs)
    candidates.push_back(("/" + archDir + archRelativeSuffix).str());
  if (!genericSuffix.empty())
    candidates.push_back(genericSuffix.str());
  return candidates;
}

static std::vector<std::string>
collectDavSimulatorLibDirMatches(llvm::ArrayRef<std::string> simulatorRoots) {
  std::vector<std::string> matches;
  for (const std::string &root : simulatorRoots) {
    if (root.empty() || !llvm::sys::fs::is_directory(root))
      continue;
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator it(root, ec), end; !ec && it != end;
         it.increment(ec)) {
      if (!llvm::sys::fs::is_directory(it->path()))
        continue;
      const std::string candidate = (llvm::Twine(it->path()) + "/lib").str();
      if (llvm::sys::fs::exists(
              (llvm::Twine(candidate) + "/libmodel_top.so").str()))
        matches.push_back(candidate);
    }
  }
  return matches;
}

static std::vector<std::string>
getDavSimulatorRoots(llvm::StringRef ascendHome, llvm::StringRef machine) {
  std::vector<std::string> suffixes = buildArchRelativeCandidates(
      "", "/simulator", machine.empty() ? std::nullopt : std::optional(machine));
  std::vector<std::string> simulatorRoots;
  simulatorRoots.reserve(suffixes.size() + 1);
  for (const std::string &suffix : suffixes)
    simulatorRoots.push_back((ascendHome + suffix).str());
  simulatorRoots.push_back((ascendHome + "/tools/simulator").str());
  return simulatorRoots;
}

static std::string
resolveDavSimulatorLibDirFromEnv(llvm::StringRef ascendHome,
                                 llvm::StringRef machine) {
  const char *version = std::getenv("ASCEND_DAV_SIM_VERSION");
  if (!version || !*version)
    return "";

  std::vector<std::string> candidates;
  for (const std::string &root : getDavSimulatorRoots(ascendHome, machine))
    candidates.push_back((llvm::Twine(root) + "/" + version + "/lib").str());

  std::string resolved = findFirstExistingPath(candidates);
  if (!resolved.empty())
    return resolved;
  return candidates.empty() ? std::string() : candidates.front();
}

} // namespace

std::string getHostCannArchDir() {
  const std::vector<std::string> candidates = getHostCannArchCandidates();
  return candidates.empty() ? std::string() : candidates.front();
}

std::string getHostCannArchDir(llvm::StringRef machine) {
  const std::vector<std::string> candidates = getCannArchCandidates(machine);
  return candidates.empty() ? std::string() : candidates.front();
}

std::string findFirstExistingPath(llvm::ArrayRef<std::string> candidates) {
  for (const std::string &candidate : candidates) {
    if (!candidate.empty() && llvm::sys::fs::exists(candidate))
      return candidate;
  }
  return "";
}

llvm::Expected<std::string> requireAscendHome() {
  std::string home = findAscendHome();
  if (!home.empty())
    return home;
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Ascend toolkit root is not configured; set ASCEND_HOME_PATH or "
      "ASCEND_TOOLKIT_HOME");
}

std::string resolveAscendHomeForTest(llvm::StringRef ascendHomeEnv,
                                     llvm::StringRef toolkitHomeEnv) {
  if (!ascendHomeEnv.empty())
    return ascendHomeEnv.str();
  if (!toolkitHomeEnv.empty())
    return toolkitHomeEnv.str();
  return "";
}

std::string resolveSocVersionForTest(llvm::StringRef explicitSocVersion,
                                     llvm::StringRef envSocVersion,
                                     llvm::StringRef fallbackSocVersion) {
  if (!explicitSocVersion.empty())
    return explicitSocVersion.str();
  if (!envSocVersion.empty())
    return envSocVersion.str();
  return fallbackSocVersion.str();
}

std::string findSocVersion() {
  if (const char *socVersion = std::getenv("SOC_VERSION"))
    if (*socVersion)
      return socVersion;
  return "";
}

std::string resolveSocVersion(llvm::StringRef explicitSocVersion,
                              llvm::StringRef fallbackSocVersion) {
  return resolveSocVersionForTest(explicitSocVersion, findSocVersion(),
                                  fallbackSocVersion);
}

std::string findAscendHome() {
  if (const char *home = std::getenv("ASCEND_HOME_PATH"))
    if (*home)
      return home;
  if (const char *toolkitHome = std::getenv("ASCEND_TOOLKIT_HOME"))
    if (*toolkitHome)
      return toolkitHome;
  return "";
}

std::string findAscendIncludeDir(llvm::StringRef ascendHome) {
  return findAscendIncludeDir(ascendHome, "");
}

std::string findAscendIncludeDir(llvm::StringRef ascendHome,
                                 llvm::StringRef machine) {
  return findUnderAscendHome(
      ascendHome,
      buildArchRelativeCandidates("/include", "/include",
                                  machine.empty() ? std::nullopt
                                                  : std::optional(machine)));
}

std::string findAscendLib64Dir(llvm::StringRef ascendHome) {
  return findAscendLib64Dir(ascendHome, "");
}

std::string findAscendLib64Dir(llvm::StringRef ascendHome,
                               llvm::StringRef machine) {
  return findUnderAscendHome(
      ascendHome,
      buildArchRelativeCandidates("/lib64", "/lib64",
                                  machine.empty() ? std::nullopt
                                                  : std::optional(machine)));
}

std::string findAscendDeviceLibDir(llvm::StringRef ascendHome) {
  return findAscendDeviceLibDir(ascendHome, "");
}

std::string findAscendDeviceLibDir(llvm::StringRef ascendHome,
                                   llvm::StringRef machine) {
  return findUnderAscendHome(ascendHome,
                             buildArchRelativeCandidates(
                                 "/lib64/device/lib64", "/lib64/device/lib64",
                                 machine.empty() ? std::nullopt
                                                 : std::optional(machine)));
}

std::string findAscendSimulatorLibDir(llvm::StringRef ascendHome,
                                      llvm::StringRef socVersion) {
  return findAscendSimulatorLibDir(ascendHome, socVersion, "");
}

std::string findAscendSimulatorLibDir(llvm::StringRef ascendHome,
                                      llvm::StringRef socVersion,
                                      llvm::StringRef machine) {
  std::vector<std::string> suffixes = buildArchRelativeCandidates(
      "", ("/simulator/" + socVersion + "/lib").str(),
      machine.empty() ? std::nullopt : std::optional(machine));
  std::vector<std::string> candidates;
  candidates.reserve(suffixes.size() + 1);
  for (const std::string &suffix : suffixes)
    candidates.push_back((ascendHome + suffix).str());
  candidates.push_back((ascendHome + "/tools/simulator/" + socVersion + "/lib")
                           .str());
  std::string resolved = findFirstExistingPath(candidates);
  if (!resolved.empty())
    return resolved;
  return candidates.front();
}

std::string findAscendDavSimulatorLibDir(llvm::StringRef ascendHome) {
  return findAscendDavSimulatorLibDir(ascendHome, "");
}

std::string findAscendDavSimulatorLibDir(llvm::StringRef ascendHome,
                                         llvm::StringRef machine) {
  if (std::string configured =
          resolveDavSimulatorLibDirFromEnv(ascendHome, machine);
      !configured.empty())
    return configured;
  std::vector<std::string> simulatorRoots =
      getDavSimulatorRoots(ascendHome, machine);
  std::vector<std::string> matches =
      collectDavSimulatorLibDirMatches(simulatorRoots);
  if (matches.size() == 1)
    return matches.front();
  return simulatorRoots.front() + "/dav";
}

llvm::Expected<std::string>
requireAscendDavSimulatorLibDir(llvm::StringRef ascendHome) {
  return requireAscendDavSimulatorLibDir(ascendHome, "");
}

llvm::Expected<std::string>
requireAscendDavSimulatorLibDir(llvm::StringRef ascendHome,
                                llvm::StringRef machine) {
  if (std::string configured =
          resolveDavSimulatorLibDirFromEnv(ascendHome, machine);
      !configured.empty()) {
    if (llvm::sys::fs::exists((llvm::Twine(configured) + "/libmodel_top.so").str()))
      return configured;
    const char *configuredVersion = std::getenv("ASCEND_DAV_SIM_VERSION");
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Configured DAV simulator version '%s' does not resolve to a valid libmodel_top.so under %s",
        configuredVersion ? configuredVersion : "", ascendHome.str().c_str());
  }
  std::vector<std::string> matches =
      collectDavSimulatorLibDirMatches(
          getDavSimulatorRoots(ascendHome, machine));
  if (matches.size() == 1)
    return matches.front();
  if (matches.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot locate DAV simulator libs under %s", ascendHome.str().c_str());
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Multiple DAV simulator lib directories found under %s; set "
      "ASCEND_DAV_SIM_VERSION",
      ascendHome.str().c_str());
}

std::string findAscendAclLibPath(llvm::StringRef ascendHome) {
  return findAscendAclLibPath(ascendHome, "");
}

std::string findAscendAclLibPath(llvm::StringRef ascendHome,
                                 llvm::StringRef machine) {
  return findAscendLib64Dir(ascendHome, machine) + "/libascendcl.so";
}

std::string findAscendRuntimeCamodelPath(llvm::StringRef ascendHome,
                                         llvm::StringRef socVersion) {
  return findAscendRuntimeCamodelPath(ascendHome, socVersion, "");
}

std::string findAscendRuntimeCamodelPath(llvm::StringRef ascendHome,
                                         llvm::StringRef socVersion,
                                         llvm::StringRef machine) {
  std::vector<std::string> candidates = {
      findAscendSimulatorLibDir(ascendHome, socVersion, machine) +
          "/libruntime_camodel.so",
      (ascendHome + "/runtime/lib64/libruntime_camodel.so").str(),
  };
  std::string resolved = findFirstExistingPath(candidates);
  if (!resolved.empty())
    return resolved;
  return candidates.front();
}

std::string findAscendTikcppDir(llvm::StringRef ascendHome) {
  std::vector<std::string> candidates = {"/toolkit/tools/tikcpp"};
  std::vector<std::string> archCandidates =
      buildArchRelativeCandidates("", "/tikcpp");
  candidates.insert(candidates.end(), archCandidates.begin(),
                    archCandidates.end());
  return findUnderAscendHome(ascendHome, candidates);
}

std::string findAscendAscDir(llvm::StringRef ascendHome) {
  return findUnderAscendHome(ascendHome,
                             buildArchRelativeCandidates("", "/asc"));
}

} // namespace mlir::runtime
