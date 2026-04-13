// tools/autotuner/autotuner_main.cpp
#include "Runtime/ArtifactCompiler.h"
#include "Runtime/ExecutionSession.h"
#include "Runtime/Executor.h"
#include "Runtime/HostRunnerGen.h"
#include "Runtime/NpyIO.h"
#include "Runtime/PathUtils.h"
#include "Runtime/ProfileTrace.h"
#include "Runtime/ProfileUtils.h"
#include "Runtime/RunManifest.h"
#include "Runtime/SimValidator.h"
#include "Runtime/TaskGraph.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <unistd.h>

using namespace mlir::runtime;
using namespace llvm;

static cl::opt<std::string> SpaceFile("space",
    cl::desc("Path to tiling_space.json"), cl::Required);
static cl::opt<std::string> KernelFile("kernel",
    cl::desc("Kernel .cpp source file (overrides kernel_file in JSON)"), cl::init(""));
static cl::opt<std::string> InputFiles("inputs",
    cl::desc("Comma-separated input .npy files"), cl::Required);
static cl::opt<std::string> ExpectedFile("expected",
    cl::desc("Expected output .npy file"), cl::Required);
static cl::opt<std::string> ShapeStr("shape",
    cl::desc("Comma-separated KEY=VALUE shape pairs: M=32,N=32"), cl::Required);
static cl::opt<std::string> OutputFile("output",
    cl::desc("Best-config JSON output path"), cl::init("best_config.json"));
static cl::opt<std::string> ArtifactRoot("artifact-root",
    cl::desc("Existing artifact root to search without recompiling"), cl::init(""));
static cl::opt<std::string> KernelKindName("kernel-kind",
    cl::desc("Kernel kind when compiling: vec, cube, or mix"), cl::init("vec"));
static cl::opt<std::string> SocVersion("soc",
    cl::desc("SoC version (overrides JSON; default: Ascend910B1)"), cl::init(""));
static cl::opt<double> Atol("atol", cl::desc("Absolute tolerance"), cl::init(1.0));
static cl::opt<double> Rtol("rtol", cl::desc("Relative tolerance"), cl::init(1e-2));
static cl::opt<std::string> ProfileOutDir("profile-out",
    cl::desc("Directory for retained profiling artifacts"), cl::init(""));
static cl::opt<bool> PerfReport("perf-report",
    cl::desc("Run msprof op simulator on best config after search (generates performance report)"),
    cl::init(false));
static cl::opt<std::string> MsprofPath("msprof",
    cl::desc("Path to msprof binary (default: auto-detect from ASCEND_HOME_PATH/tools/profiler/bin/msprof)"),
    cl::init(""));
static cl::opt<std::string> PerfReportOutDir("perf-report-out",
    cl::desc("Output directory for perf-report files (default: ./perf_out)"),
    cl::init("perf_out"));

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::vector<std::string> splitComma(const std::string& s) {
  std::vector<std::string> parts;
  std::istringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) parts.push_back(tok);
  return parts;
}

static std::map<std::string, int64_t> parseKV(const std::string& s) {
  std::map<std::string, int64_t> m;
  for (auto& tok : splitComma(s)) {
    auto eq = tok.find('=');
    if (eq == std::string::npos) continue;
    int64_t val = 0;
    llvm::StringRef(tok.substr(eq + 1)).getAsInteger(10, val);
    m[tok.substr(0, eq)] = val;
  }
  return m;
}

// Evaluate block_dim_expr: supports "ceil(X/Y)" and "X/Y"
static int64_t evalBlockDimExpr(const std::string& expr,
                                 const std::map<std::string, int64_t>& vars) {
  bool is_ceil = false;
  std::string inner;
  {
    std::string e = expr;
    e.erase(std::remove(e.begin(), e.end(), ' '), e.end());
    if (e.size() > 5 && e.substr(0, 5) == "ceil(" && e.back() == ')') {
      inner = e.substr(5, e.size() - 6);
      is_ceil = true;
    } else {
      inner = e;
    }
  }
  auto slash = inner.find('/');
  if (slash == std::string::npos) {
    auto it = vars.find(inner);
    return it != vars.end() ? it->second : 1;
  }
  std::string lhs = inner.substr(0, slash);
  std::string rhs = inner.substr(slash + 1);
  auto lookup = [&](const std::string& name) -> int64_t {
    auto it = vars.find(name);
    if (it != vars.end()) return it->second;
    int64_t v = 1;
    if (llvm::StringRef(name).getAsInteger(10, v)) {
      llvm::errs() << "Warning: unrecognized token in block_dim_expr: '" << name << "', treating as 1\n";
      return 1;
    }
    return v;
  };
  int64_t a = lookup(lhs), b = lookup(rhs);
  if (b == 0) return 1;
  if (is_ceil) return (a + b - 1) / b;
  return a / b;
}

static std::vector<uint8_t> packTiling(
    const std::vector<std::pair<std::string, int64_t>>& params,
    const std::vector<std::string>& types) {
  std::vector<uint8_t> bytes;
  for (size_t i = 0; i < params.size(); ++i) {
    std::string type = i < types.size() ? types[i] : "int64";
    int64_t val = params[i].second;
    if (type == "int32" || type == "int32_t") {
      int32_t v = static_cast<int32_t>(val);
      uint8_t buf[4]; std::memcpy(buf, &v, 4);
      bytes.insert(bytes.end(), buf, buf + 4);
    } else {
      uint8_t buf[8]; std::memcpy(buf, &val, 8);
      bytes.insert(bytes.end(), buf, buf + 8);
    }
  }
  return bytes;
}

static llvm::StringRef kernelKindToString(KernelKind kind) {
  switch (kind) {
  case KernelKind::Vec:
    return "vec";
  case KernelKind::Cube:
    return "cube";
  case KernelKind::Mix:
    return "mix";
  }
  return "vec";
}

static llvm::Expected<KernelKind> parseKernelKind(llvm::StringRef name) {
  if (name == "vec")
    return KernelKind::Vec;
  if (name == "cube")
    return KernelKind::Cube;
  if (name == "mix")
    return KernelKind::Mix;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported kernel kind: %s",
                                 name.str().c_str());
}

static llvm::Expected<KernelKind> parseManifestKernelKind(llvm::StringRef name) {
  if (name.empty())
    return KernelKind::Mix;
  return parseKernelKind(name);
}

static std::string defaultKernelName(llvm::StringRef kernelFile) {
  if (kernelFile.empty())
    return "";
  return llvm::sys::path::stem(kernelFile).str();
}

static std::map<std::string, std::string> readManifest(const std::string &path) {
  std::map<std::string, std::string> out;
  auto bufferOr = llvm::MemoryBuffer::getFile(path, false);
  if (!bufferOr)
    return out;

  llvm::SmallVector<llvm::StringRef> lines;
  (*bufferOr)->getBuffer().split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (line.empty() || line.starts_with("#"))
      continue;
    size_t split = line.find('=');
    if (split == llvm::StringRef::npos)
      continue;
    out.emplace(line.substr(0, split).str(), line.substr(split + 1).str());
  }
  return out;
}

static llvm::Expected<std::string> requireManifestValue(
    const std::map<std::string, std::string> &manifest, llvm::StringRef key) {
  auto it = manifest.find(key.str());
  if (it == manifest.end() || it->second.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "manifest is missing required field: %s",
                                   key.str().c_str());
  }
  return it->second;
}

static std::string resolveArtifactPath(llvm::StringRef artifactRoot,
                                       llvm::StringRef maybeRelativePath) {
  if (maybeRelativePath.empty())
    return "";
  if (llvm::sys::path::is_absolute(maybeRelativePath))
    return maybeRelativePath.str();

  llvm::SmallString<256> resolved(artifactRoot);
  llvm::sys::path::append(resolved, maybeRelativePath);
  return resolved.str().str();
}

static llvm::Expected<std::string> locateManifestPath(llvm::StringRef artifactRoot) {
  llvm::SmallVector<llvm::SmallString<256>, 3> candidates;
  candidates.emplace_back(artifactRoot);
  llvm::sys::path::append(candidates.back(), "out", "manifest.txt");
  candidates.emplace_back(artifactRoot);
  llvm::sys::path::append(candidates.back(), "mix-artifact.txt");
  candidates.emplace_back(artifactRoot);
  llvm::sys::path::append(candidates.back(), "out", "mix-artifact.txt");

  for (const llvm::SmallString<256> &candidate : candidates) {
    if (llvm::sys::fs::exists(candidate))
      return candidate.str().str();
  }

  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "artifact root does not contain a supported manifest (expected out/manifest.txt, mix-artifact.txt, or out/mix-artifact.txt)");
}

static llvm::Expected<KernelArtifact> loadArtifactFromRoot(llvm::StringRef artifactRootInput) {
  llvm::SmallString<256> artifactRoot(artifactRootInput);
  llvm::sys::fs::make_absolute(artifactRoot);

  llvm::sys::fs::file_status status;
  if (auto ec = llvm::sys::fs::status(artifactRoot, status))
    return llvm::createStringError(ec, "cannot access artifact root: %s",
                                   artifactRoot.c_str());
  if (!llvm::sys::fs::is_directory(status)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "artifact root is not a directory: %s",
                                   artifactRoot.c_str());
  }

  auto manifestPathOr = locateManifestPath(artifactRoot.str());
  if (!manifestPathOr)
    return manifestPathOr.takeError();

  const std::string manifestPath = *manifestPathOr;
  const auto manifest = readManifest(manifestPath);
  if (manifest.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "manifest is empty or unreadable: %s",
                                   manifestPath.c_str());
  }

  auto kernelNameOr = requireManifestValue(manifest, "kernel_name");
  if (!kernelNameOr)
    return kernelNameOr.takeError();
  auto socVersionOr = requireManifestValue(manifest, "soc_version");
  if (!socVersionOr)
    return socVersionOr.takeError();

  KernelArtifact artifact;
  artifact.kernelName = *kernelNameOr;
  auto kernelKindIt = manifest.find("kernel_kind");
  auto parsedKernelKindOr = parseManifestKernelKind(
      kernelKindIt != manifest.end() ? kernelKindIt->second : "");
  if (!parsedKernelKindOr)
    return parsedKernelKindOr.takeError();
  artifact.kernelKind = *parsedKernelKindOr;
  artifact.mixResourceType = MixResourceType::Unknown;
  artifact.socVersion = *socVersionOr;
  artifact.artifactRoot = artifactRoot.str().str();

  auto manifestPathIt = manifest.find("manifest_path");
  if (manifestPathIt != manifest.end() && !manifestPathIt->second.empty()) {
    artifact.manifestPath =
        resolveArtifactPath(artifact.artifactRoot, manifestPathIt->second);
    if (!llvm::sys::fs::exists(artifact.manifestPath)) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "manifest_path from manifest does not exist: %s",
          artifact.manifestPath.c_str());
    }
  } else {
    artifact.manifestPath = manifestPath;
  }

  auto kernelSoIt = manifest.find("kernel_so_path");
  if (kernelSoIt != manifest.end() && !kernelSoIt->second.empty())
    artifact.packedSharedObjectPath =
        resolveArtifactPath(artifact.artifactRoot, kernelSoIt->second);

  auto deviceObjectIt = manifest.find("device_object_path");
  auto deviceBinaryIt = manifest.find("device_binary_path");
  if (deviceBinaryIt != manifest.end() && !deviceBinaryIt->second.empty()) {
    artifact.deviceBinaryPath =
        resolveArtifactPath(artifact.artifactRoot, deviceBinaryIt->second);
  } else if (deviceObjectIt != manifest.end() && !deviceObjectIt->second.empty()) {
    artifact.deviceBinaryPath =
        resolveArtifactPath(artifact.artifactRoot, deviceObjectIt->second);
  } else if (!artifact.packedSharedObjectPath.empty()) {
    artifact.deviceBinaryPath = artifact.packedSharedObjectPath;
  }

  return artifact;
}

// ── Data structures ───────────────────────────────────────────────────────────

struct TilingParam {
  std::string name;
  std::string type;
  bool        fixed = false;
  std::string shape_key;
  std::vector<int64_t> values;
};

struct TilingSpace {
  std::string kernel_name;
  std::string kernel_file;
  std::string kernel_type = "vec";   // "vec" | "cube" | "mix"; default vec
  std::string soc;
  std::string block_dim_expr;
  std::vector<TilingParam> params;
};

struct SearchInputs {
  std::vector<std::string> inputFiles;
  std::string expectedFile;
  std::map<std::string, int64_t> shape;
  double atol = 1.0;
  double rtol = 1e-2;
};

struct CandidateExecutionSpec {
  std::vector<std::pair<std::string, int64_t>> params;
  int64_t blockDim = 1;
  std::string actualOutputPath;
  std::string profileOutputDir;
};

static llvm::Expected<TilingSpace> loadTilingSpace(const std::string& path) {
  auto buf = llvm::MemoryBuffer::getFile(path);
  if (!buf)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot open %s", path.c_str());
  auto json = llvm::json::parse((*buf)->getBuffer());
  if (!json)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "JSON parse error in %s", path.c_str());

  TilingSpace ts;
  auto* obj = json->getAsObject();
  if (!obj)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "JSON root is not an object in %s", path.c_str());

  // getString returns std::optional<llvm::StringRef>
  if (auto v = obj->getString("kernel"))         ts.kernel_name    = v->str();
  if (auto v = obj->getString("kernel_file"))    ts.kernel_file    = v->str();
  if (auto v = obj->getString("kernel_type"))    ts.kernel_type    = v->str();
  if (auto v = obj->getString("soc"))            ts.soc            = v->str();
  if (auto v = obj->getString("block_dim_expr")) ts.block_dim_expr = v->str();

  auto* params = obj->getArray("tiling_params");
  if (!params)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Missing tiling_params in %s", path.c_str());

  for (auto& pv : *params) {
    auto* po = pv.getAsObject();
    if (!po) continue;
    TilingParam p;
    if (auto v = po->getString("name"))      p.name = v->str();
    if (auto v = po->getString("type"))      p.type = v->str();
    else                                     p.type = "int64";
    if (auto v = po->getBoolean("fixed"))    p.fixed = *v;
    if (auto v = po->getString("shape_key")) p.shape_key = v->str();

    if (!p.fixed) {
      if (auto* arr = po->getArray("values")) {
        for (auto& av : *arr)
          if (auto iv = av.getAsInteger()) p.values.push_back(*iv);
      }
      if (p.values.empty()) {
        int64_t mn = 0, mx = 0, st = 1;
        if (auto v = po->getInteger("min"))  mn = *v;
        if (auto v = po->getInteger("max"))  mx = *v;
        if (auto v = po->getInteger("step")) st = *v;
        if (st > 0)
          for (int64_t x = mn; x <= mx; x += st) p.values.push_back(x);
      }
    }
    ts.params.push_back(p);
  }
  return ts;
}

// ── Search ────────────────────────────────────────────────────────────────────

struct SearchResult {
  std::vector<std::pair<std::string, int64_t>> config;
  int64_t   block_dim    = 1;
  int64_t   cycle_count  = -1;
  double    max_abs_diff = 0.0;
  bool      passed       = false;
};

static llvm::Expected<KernelArtifact> prepareArtifact(const TilingSpace &space) {
  if (!ArtifactRoot.empty())
    return loadArtifactFromRoot(ArtifactRoot);

  std::string kernelPath = !KernelFile.empty() ? KernelFile.getValue()
                                               : space.kernel_file;
  if (kernelPath.empty()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "kernel file not specified (--kernel, --artifact-root, or kernel_file in JSON)");
  }
  if (KernelFile.empty() && !llvm::sys::path::is_absolute(kernelPath)) {
    llvm::SmallString<256> base(SpaceFile.getValue());
    llvm::sys::path::remove_filename(base);
    llvm::sys::path::append(base, kernelPath);
    kernelPath = base.str().str();
  }

  std::string kernelKindName =
      KernelKindName.getNumOccurrences() > 0 ? KernelKindName.getValue()
                                             : space.kernel_type;
  if (kernelKindName.empty())
    kernelKindName = "vec";
  auto kernelKindOr = parseKernelKind(kernelKindName);
  if (!kernelKindOr)
    return kernelKindOr.takeError();

  std::string soc = !SocVersion.empty() ? SocVersion.getValue() : space.soc;
  if (soc.empty())
    soc = "Ascend910B1";

  std::string effectiveKernelName = space.kernel_name.empty()
                                        ? defaultKernelName(kernelPath)
                                        : space.kernel_name;
  if (effectiveKernelName.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "kernel name could not be derived");
  }

  llvm::SmallString<256> buildDir;
  if (llvm::sys::fs::createUniqueDirectory("autotuner_build", buildDir)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot create temp build dir");
  }

  ArtifactCompileRequest request;
  request.kernelSource = kernelPath;
  request.kernelName = effectiveKernelName;
  request.kernelKind = *kernelKindOr;
  request.socVersion = soc;
  request.outputDir = buildDir.str().str();
  ArtifactCompiler compiler;
  return compiler.compile(request);
}

static llvm::Error writeBestConfigJson(const std::string &path,
                                      const TilingSpace &space,
                                      const KernelArtifact &artifact,
                                      const SearchResult &best,
                                      const std::map<std::string, int64_t> &shape) {
  llvm::json::Object root;
  root["kernel_name"] = artifact.kernelName;
  root["kernel_type"] = std::string(kernelKindToString(artifact.kernelKind));
  root["soc"] = artifact.socVersion;
  root["artifact_root"] = artifact.artifactRoot;
  root["manifest_path"] = artifact.manifestPath;
  root["device_binary_path"] = artifact.deviceBinaryPath;
  if (!ProfileOutDir.empty())
    root["profile_out"] = ProfileOutDir.getValue();
  root["block_dim"] = best.block_dim;
  root["cycle_count"] = best.cycle_count;
  root["score"] = best.cycle_count;
  root["max_abs_diff"] = best.max_abs_diff;
  root["passed"] = best.passed;
  root["kernel_file"] = space.kernel_file;

  llvm::json::Object config;
  for (const auto &kv : best.config)
    config[kv.first] = kv.second;
  root["config"] = std::move(config);

  llvm::json::Object shapeObj;
  for (const auto &kv : shape)
    shapeObj[kv.first] = kv.second;
  root["shape"] = std::move(shapeObj);

  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec)
    return llvm::createStringError(ec, "cannot write best-config JSON");
  llvm::json::OStream jos(os, /*IndentSize=*/2);
  jos.value(llvm::json::Value(std::move(root)));
  os << "\n";
  return llvm::Error::success();
}

static void enumerateCombos(
    const std::vector<TilingParam>& search_vars,
    size_t idx,
    std::vector<int64_t>& current,
    std::vector<std::vector<int64_t>>& results) {
  if (idx == search_vars.size()) {
    results.push_back(current);
    return;
  }
  for (int64_t v : search_vars[idx].values) {
    current[idx] = v;
    enumerateCombos(search_vars, idx + 1, current, results);
  }
}

static std::vector<SearchResult> runSearch(
    const TilingSpace& ts,
    const std::map<std::string, int64_t>& shape,
    const KernelArtifact& artifact,
    RunArgs& args_template,
    const std::vector<NDArray>& expected,
    double atol, double rtol,
    std::string* out_binary_path = nullptr) {

  std::vector<TilingParam> search_vars;
  for (auto& p : ts.params)
    if (!p.fixed && !p.values.empty()) search_vars.push_back(p);

  std::vector<std::vector<int64_t>> combos;
  std::vector<int64_t> cur(search_vars.size(), 0);
  enumerateCombos(search_vars, 0, cur, combos);

  int total = static_cast<int>(combos.size());
  std::vector<SearchResult> results;
  std::string binary_path = !artifact.deviceBinaryPath.empty()
                                ? artifact.deviceBinaryPath
                                : artifact.packedSharedObjectPath;
  if (binary_path.empty()) {
    llvm::errs() << "Error: artifact does not contain a binary path\n";
    return results;
  }
  if (out_binary_path) *out_binary_path = binary_path;

  // Initialize executor once and reuse across all configs.
  Executor executor;
  if (auto err = executor.Initialize()) {
    llvm::errs() << "Error: executor init failed: " << llvm::toString(std::move(err)) << "\n";
    return results;
  }

  // Register binary+function once — all configs use the same ELF binary
  // (tiling is passed as kernel args, not compiled-in). Registering once
  // avoids rc=507000 from the simulator when the same stub pointer is
  // re-registered under a new binary handle on subsequent RunFile() calls.
  auto handle_or = executor.RegisterBinary(binary_path,
                                           artifact.kernelName.empty()
                                               ? ts.kernel_name
                                               : artifact.kernelName);
  if (!handle_or) {
    llvm::errs() << "Error: register binary failed: "
                 << llvm::toString(handle_or.takeError()) << "\n";
    return results;
  }
  void* func_handle = *handle_or;

  SimValidator validator;

  for (int ci = 0; ci < total; ++ci) {
    std::map<std::string, int64_t> vars = shape;
    for (size_t si = 0; si < search_vars.size(); ++si)
      vars[search_vars[si].name] = combos[ci][si];

    std::vector<std::pair<std::string, int64_t>> param_vals;
    std::vector<std::string> param_types;
    bool param_error = false;
    for (auto& p : ts.params) {
      int64_t val = 0;
      if (p.fixed) {
        if (p.shape_key.empty()) {
          llvm::errs() << "Error: fixed param '" << p.name
                       << "' has no shape_key in tiling_space.json\n";
          param_error = true;
          break;
        }
        auto it = shape.find(p.shape_key);
        if (it == shape.end()) {
          llvm::errs() << "Error: shape key '" << p.shape_key
                       << "' not found in --shape (needed by param '" << p.name << "')\n";
          param_error = true;
          break;
        }
        val = it->second;
      } else {
        auto it = vars.find(p.name);
        val = it != vars.end() ? it->second : 0;
      }
      param_vals.push_back({p.name, val});
      param_types.push_back(p.type);
    }
    if (param_error) break;

    int64_t block_dim = ts.block_dim_expr.empty() ? 1
                        : evalBlockDimExpr(ts.block_dim_expr, vars);

    llvm::outs() << "[" << (ci + 1) << "/" << total << "]";
    for (auto& kv : param_vals)
      if (kv.first.compare(0, 4, "dim_") != 0)
        llvm::outs() << " " << kv.first << "=" << kv.second;
    llvm::outs() << " block_dim=" << block_dim << " ...";
    llvm::outs().flush();

    std::memset(args_template.outputs[0].data,
                0, args_template.outputs[0].nbytes());

    args_template.tiling    = packTiling(param_vals, param_types);
    args_template.block_dim = static_cast<int>(block_dim);

    auto res = validator.ValidateBinary(func_handle, executor, args_template,
                                        expected, atol, rtol);

    SearchResult sr;
    sr.config       = param_vals;
    sr.block_dim    = block_dim;
    sr.cycle_count  = res.cycle_count;
    sr.max_abs_diff = res.max_abs_diff;
    sr.passed       = res.passed;

    if (res.passed)
      llvm::outs() << " PASS  cycles=" << res.cycle_count
                   << " max_diff=" << res.max_abs_diff << "\n";
    else
      llvm::outs() << " FAIL  " << res.error_msg << "\n";

    results.push_back(sr);
  }
  return results;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv, "AscendC AutoTuner\n");

  // Note: _Exit() calls below bypass destructors to avoid simulator background
  // thread race on process exit (same reason as sim-validator tool).
  auto ts_or = loadTilingSpace(SpaceFile);
  if (!ts_or) {
    llvm::errs() << "Error: " << llvm::toString(ts_or.takeError()) << "\n";
    _Exit(1);
  }
  TilingSpace ts = std::move(*ts_or);

  auto artifactOr = prepareArtifact(ts);
  if (!artifactOr) {
    llvm::errs() << "Error: " << llvm::toString(artifactOr.takeError()) << "\n";
    _Exit(1);
  }
  KernelArtifact artifact = std::move(*artifactOr);
  ts.kernel_name = artifact.kernelName.empty() ? ts.kernel_name : artifact.kernelName;
  ts.kernel_type = std::string(kernelKindToString(artifact.kernelKind));
  ts.soc = artifact.socVersion;

  auto shape = parseKV(ShapeStr);

  // Load inputs
  std::vector<NDArray> inputs;
  for (auto& path : splitComma(InputFiles)) {
    auto arr_or = LoadNpy(path);
    if (!arr_or) {
      for (auto& inp : inputs) delete[] static_cast<uint8_t*>(inp.data);
      llvm::errs() << "Error loading " << path << ": "
                   << llvm::toString(arr_or.takeError()) << "\n";
      _Exit(1);
    }
    inputs.push_back(std::move(*arr_or));
  }

  // Load expected
  auto exp_or = LoadNpy(ExpectedFile);
  if (!exp_or) {
    for (auto& inp : inputs) delete[] static_cast<uint8_t*>(inp.data);
    llvm::errs() << "Error loading expected: " << llvm::toString(exp_or.takeError()) << "\n";
    _Exit(1);
  }
  std::vector<NDArray> expected_arrs;
  expected_arrs.push_back(std::move(*exp_or));

  // Pre-alloc output buffer (shape/dtype from expected)
  NDArray out_buf;
  out_buf.shape = expected_arrs[0].shape;
  out_buf.dtype = expected_arrs[0].dtype;
  out_buf.data  = new uint8_t[out_buf.nbytes()]();

  RunArgs args_tmpl;
  args_tmpl.inputs  = std::move(inputs);
  args_tmpl.outputs.push_back(std::move(out_buf));

  llvm::outs() << "Searching " << ts.kernel_name << " on " << ts.soc << "\n";
  std::string best_binary_path;
  auto results = runSearch(ts, shape, artifact,
                            args_tmpl, expected_arrs, Atol, Rtol,
                            &best_binary_path);

  // Free allocations
  delete[] static_cast<uint8_t*>(out_buf.data);
  for (auto& inp : inputs) delete[] static_cast<uint8_t*>(inp.data);
  delete[] static_cast<uint8_t*>(expected_arrs[0].data);

  // Filter PASS results
  std::vector<SearchResult*> passed;
  for (auto& r : results) if (r.passed) passed.push_back(&r);

  if (passed.empty()) {
    llvm::errs() << "Error: no configuration passed validation\n";
    _Exit(1);
  }

  // Sort by cycle_count ascending; -1 (unavailable) goes last
  std::sort(passed.begin(), passed.end(), [](SearchResult* a, SearchResult* b) {
    if (a->cycle_count < 0) return false;
    if (b->cycle_count < 0) return true;
    return a->cycle_count < b->cycle_count;
  });

  SearchResult& best = *passed[0];
  llvm::outs() << "\nBest config: cycles=" << best.cycle_count
               << " max_diff=" << best.max_abs_diff << "\n";
  for (auto& kv : best.config)
    if (kv.first.compare(0, 4, "dim_") != 0)
      llvm::outs() << "  " << kv.first << "=" << kv.second << "\n";

  if (auto err = writeBestConfigJson(OutputFile, ts, artifact, best, shape)) {
    llvm::errs() << "Error: " << llvm::toString(std::move(err)) << "\n";
    _Exit(1);
  }

  // ── Optional: perf report via msprof op simulator ───────────────────────────
  if (PerfReport) {
    // Step 1: generate runner executable via HostRunnerGen
    HostRunnerGen::Config hcfg;
    hcfg.kernel_name  = ts.kernel_name;
    hcfg.kernel_type  = ts.kernel_type;
    hcfg.soc_version  = ts.soc;
    hcfg.num_inputs   = static_cast<int>(splitComma(InputFiles).size());
    hcfg.num_outputs  = 1;
    for (auto& p : ts.params)
      hcfg.tiling_layout.push_back(p.type);

    // Use same build dir as the binary (derive from best_binary_path)
    std::string runner_dir = best_binary_path.empty() ? "."
        : llvm::sys::path::parent_path(best_binary_path).str();

    HostRunnerGen gen;
    auto runner_or = gen.Generate(hcfg, runner_dir);
    if (!runner_or) {
      llvm::errs() << "Warning: --perf-report: runner generation failed: "
                   << llvm::toString(runner_or.takeError()) << "\n";
      goto perf_done;
    }
    // runner_path declared AFTER the goto above — no jump-over-initialization issue
    {
      std::string runner_path = *runner_or;

      // Step 2: build tiling_params and tiling_layout strings for runner CLI
      std::string tiling_params_str;
      for (size_t i = 0; i < best.config.size(); ++i) {
        if (i) tiling_params_str += ",";
        tiling_params_str += best.config[i].first + "=" +
                             std::to_string(best.config[i].second);
      }
      std::string tiling_layout_str;
      for (size_t i = 0; i < ts.params.size(); ++i) {
        if (i) tiling_layout_str += ",";
        tiling_layout_str += ts.params[i].type;
      }

      // Step 3: locate msprof binary
      std::string msprof = MsprofPath;
      if (msprof.empty()) {
        auto ascendHomeOr = requireAscendHome();
        if (!ascendHomeOr) {
          llvm::errs() << "Warning: --perf-report requires ASCEND_HOME_PATH or "
                          "ASCEND_TOOLKIT_HOME, or pass --msprof=<path>\n";
          goto perf_done;
        }
        msprof = *ascendHomeOr + "/tools/profiler/bin/msprof";
      }
      if (!llvm::sys::fs::exists(msprof)) {
        llvm::errs() << "Warning: --perf-report: msprof not found at: " << msprof << "\n"
                     << "  Set ASCEND_HOME_PATH or use --msprof=<path>\n";
        goto perf_done;
      }

      // Step 4: create output dir and invoke msprof op simulator.
      // msprof writes report files to cwd, so we cd into the output dir first.
      llvm::SmallString<256> out_abs(PerfReportOutDir.getValue());
      llvm::sys::fs::make_absolute(out_abs);
      llvm::sys::fs::create_directories(out_abs);

      // cd into output dir so msprof drops files there; use absolute paths for
      // runner and bin so they resolve correctly from the new cwd.
      llvm::SmallString<256> runner_abs(runner_path);
      llvm::sys::fs::make_absolute(runner_abs);
      llvm::SmallString<256> bin_abs(best_binary_path);
      llvm::sys::fs::make_absolute(bin_abs);

      std::string cmd = "cd \"" + out_abs.str().str() + "\" && "
          + "\"" + msprof + "\""
          + " op simulator --soc-version=" + ts.soc
          + " \"" + runner_abs.str().str() + "\""
          + " --bin \"" + bin_abs.str().str() + "\""
          + " --tiling-params \"" + tiling_params_str + "\""
          + " --tiling-layout \"" + tiling_layout_str + "\""
          + " --inputs \"" + InputFiles.getValue() + "\""
          + " --output /dev/null"
          + " --block-dim " + std::to_string(best.block_dim);

      llvm::outs() << "\nRunning msprof op simulator ...\n  " << cmd << "\n";
      llvm::outs().flush();
      int rc = std::system(cmd.c_str());
      if (rc != 0)
        llvm::errs() << "Warning: msprof exited with code " << rc << "\n";
      else
        llvm::outs() << "Perf report written to: " << out_abs.str() << "\n";
    }
  }
  perf_done:;

  llvm::outs().flush();
  llvm::errs().flush();
  // Use _Exit (not return/exit) to skip C++ destructors and atexit handlers:
  // SimValidator invokes libruntime_camodel which leaves background simulator
  // threads running; normal exit() races those threads and segfaults.
  _Exit(0);
}
