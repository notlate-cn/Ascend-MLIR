// tools/autotuner/autotuner_main.cpp
#include "Runtime/ArtifactCompiler.h"
#include "Runtime/ExecutionSession.h"
#include "Runtime/NpyIO.h"
#include "Runtime/ProfileTrace.h"
#include "Runtime/ProfileUtils.h"
#include "Runtime/TaskGraph.h"
#include "Runtime/RuntimeSessionRequestBuilder.h"
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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <limits>
#include <sstream>

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

// Evaluate a block-dim expression.  Grammar (produced by TilePlanGen, mirrors
// SymExpr::emitC plus a ceil() wrapper):
//   expr ::= name | int | '(' expr op expr ')' | 'ceil(' expr '/' expr ')'
//   op   ::= + | - | * | /                       (/ is integer division)
// `name` is a tiling-param ("XBLOCK") or a shape key ("arg0_dim2"); unknown
// names default to 1 with a warning.  Whitespace is stripped first.
static int64_t evalBlockExpr(llvm::StringRef e,
                             const std::map<std::string, int64_t>& vars) {
  e = e.trim();
  if (e.empty()) return 1;

  // ceil( A / B ) -- split on the *last* top-level '/'.
  if (e.starts_with("ceil(") && e.back() == ')') {
    llvm::StringRef inner = e.drop_front(5).drop_back(1);
    int depth = 0;
    size_t slash = llvm::StringRef::npos;
    for (size_t i = 0; i < inner.size(); ++i) {
      char c = inner[i];
      if (c == '(') ++depth;
      else if (c == ')') --depth;
      else if (c == '/' && depth == 0) slash = i;
    }
    if (slash == llvm::StringRef::npos) return evalBlockExpr(inner, vars);
    int64_t a = evalBlockExpr(inner.substr(0, slash), vars);
    int64_t b = evalBlockExpr(inner.substr(slash + 1), vars);
    return b == 0 ? 1 : (a + b - 1) / b;
  }

  // ( A op B ) -- emitC fully parenthesizes, so exactly one top-level op.
  if (e.front() == '(' && e.back() == ')') {
    llvm::StringRef inner = e.drop_front(1).drop_back(1);
    int depth = 0;
    for (size_t i = 0; i < inner.size(); ++i) {
      char c = inner[i];
      if (c == '(') { ++depth; continue; }
      if (c == ')') { --depth; continue; }
      if (depth != 0 || i == 0) continue;
      if (c != '+' && c != '-' && c != '*' && c != '/') continue;
      char p = inner[i - 1];
      if (p == '+' || p == '-' || p == '*' || p == '/' || p == '(') continue; // unary sign
      int64_t a = evalBlockExpr(inner.substr(0, i), vars);
      int64_t b = evalBlockExpr(inner.substr(i + 1), vars);
      switch (c) {
      case '+': return a + b;
      case '-': return a - b;
      case '*': return a * b;
      case '/': return b == 0 ? 1 : a / b;
      }
    }
    return evalBlockExpr(inner, vars); // redundant parens around a leaf
  }

  // leaf: integer literal or variable name.
  int64_t v;
  if (!e.getAsInteger(10, v)) return v;
  auto it = vars.find(e.str());
  if (it != vars.end()) return it->second;
  llvm::errs() << "Warning: unrecognized token in block_dim_expr: '" << e
               << "', treating as 1\n";
  return 1;
}

static int64_t evalBlockDimExpr(const std::string& expr,
                                const std::map<std::string, int64_t>& vars) {
  std::string e = expr;
  e.erase(std::remove(e.begin(), e.end(), ' '), e.end());
  return evalBlockExpr(e, vars);
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

static std::string defaultKernelName(llvm::StringRef kernelFile) {
  if (kernelFile.empty())
    return "";
  return llvm::sys::path::stem(kernelFile).str();
}

static std::string resolveAbsolutePath(llvm::StringRef path) {
  if (path.empty())
    return "";
  llvm::SmallString<256> absolute(path);
  llvm::sys::fs::make_absolute(absolute);
  return absolute.str().str();
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
  std::vector<std::string> paramTypes;
  int64_t blockDim = 1;
  std::string actualOutputPath;
  std::string profileOutputDir;
};

static llvm::Expected<std::vector<TensorBinding>>
loadInputBindings(const std::vector<std::string> &inputFiles) {
  std::vector<TensorBinding> bindings;
  bindings.reserve(inputFiles.size());
  for (size_t index = 0; index < inputFiles.size(); ++index) {
    auto arrayOr = LoadNpy(inputFiles[index]);
    if (!arrayOr)
      return arrayOr.takeError();

    TensorBinding binding;
    binding.name = "input" + std::to_string(index);
    binding.sourceKind = BindingSourceKind::ExternalFile;
    binding.path = resolveAbsolutePath(inputFiles[index]);
    binding.shape = arrayOr->shape;
    binding.dtype = arrayOr->dtype;
    bindings.push_back(std::move(binding));
  }
  return bindings;
}

static llvm::Expected<TensorBinding>
buildOutputBinding(const std::string &expectedFile,
                   const std::string &actualOutputPath) {
  auto arrayOr = LoadNpy(expectedFile);
  if (!arrayOr)
    return arrayOr.takeError();

  TensorBinding binding;
  binding.name = "output";
  binding.sourceKind = BindingSourceKind::ExternalFile;
  binding.path = resolveAbsolutePath(actualOutputPath);
  binding.shape = arrayOr->shape;
  binding.dtype = arrayOr->dtype;
  return binding;
}

static llvm::Expected<TensorBinding>
buildExpectedOutputBinding(const std::string &expectedFile) {
  auto arrayOr = LoadNpy(expectedFile);
  if (!arrayOr)
    return arrayOr.takeError();

  TensorBinding binding;
  binding.name = "expected_output";
  binding.sourceKind = BindingSourceKind::ExternalFile;
  binding.path = resolveAbsolutePath(expectedFile);
  binding.shape = arrayOr->shape;
  binding.dtype = arrayOr->dtype;
  return binding;
}

static std::optional<TilingBinding>
buildTilingBinding(const std::vector<uint8_t> &tilingBytes,
                   llvm::StringRef workingDir) {
  if (tilingBytes.empty())
    return std::nullopt;
  if (workingDir.empty())
    return std::nullopt;

  llvm::SmallString<256> tilingPath(workingDir);
  llvm::sys::path::append(tilingPath, "tiling.bin");

  if (auto ec = llvm::sys::fs::create_directories(workingDir)) {
    llvm::errs() << "Warning: failed to create tiling directory "
                 << workingDir << ": " << ec.message() << "\n";
    return std::nullopt;
  }

  std::error_code ec;
  llvm::raw_fd_ostream os(tilingPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "Warning: failed to write tiling binary "
                 << tilingPath << ": " << ec.message() << "\n";
    return std::nullopt;
  }
  os.write(reinterpret_cast<const char *>(tilingBytes.data()),
           static_cast<std::streamsize>(tilingBytes.size()));
  os.flush();
  if (os.has_error()) {
    llvm::errs() << "Warning: failed to flush tiling binary "
                 << tilingPath << "\n";
    return std::nullopt;
  }

  TilingBinding binding;
  binding.binaryPath = tilingPath.str().str();
  return binding;
}

static bool isRuntimeScoreKey(llvm::StringRef key) {
  return key == "cycle_count" || key == "kernel_cycle_count" ||
         key == "total_cycles" || key == "cycles" || key == "cycle" ||
         key == "total_ticks" || key == "kernel_total_ticks" ||
         key == "kernal_total_ticks" || key == "ticks" ||
         key == "duration" || key == "dur" ||
         key == "task_duration" || key == "task_duration_us" ||
         key == "task_duration_usec" || key == "elapsed_cycles" ||
         key == "elapsed";
}

static void collectRuntimeScoreCandidates(const llvm::json::Value &value,
                                         int64_t &bestScore,
                                         bool &foundAny) {
  if (const auto *object = value.getAsObject()) {
    for (const auto &entry : *object) {
      if (!isRuntimeScoreKey(entry.first))
        continue;
      if (auto number = entry.second.getAsInteger()) {
        if (*number < 0)
          continue;
        foundAny = true;
        bestScore = std::max(bestScore, *number);
      }
    }
    for (const auto &entry : *object)
      collectRuntimeScoreCandidates(entry.second, bestScore, foundAny);
    return;
  }

  if (const auto *array = value.getAsArray()) {
    for (const auto &element : *array)
      collectRuntimeScoreCandidates(element, bestScore, foundAny);
  }
}

static llvm::Expected<int64_t>
extractRuntimeScore(const ProfileTrace &trace) {
  const std::vector<std::string> artifactPaths = trace.profileArtifactPaths();
  if (artifactPaths.empty()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "runtime profiling did not produce a profile artifact");
  }

  const std::string profilePath = resolveAbsolutePath(artifactPaths.front());
  auto bufferOr = llvm::MemoryBuffer::getFile(profilePath, false);
  if (!bufferOr) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "cannot read runtime profile artifact: %s", profilePath.c_str());
  }

  auto parsedOr = llvm::json::parse((*bufferOr)->getBuffer());
  if (!parsedOr) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "failed to parse runtime profile JSON: %s", profilePath.c_str());
  }

  if (const auto *object = parsedOr->getAsObject()) {
    if (auto score = object->getInteger("score"))
      return *score;
    if (auto cycles = object->getInteger("cycle_count"))
      return *cycles;
  }

  int64_t bestScore = std::numeric_limits<int64_t>::min();
  bool foundAny = false;
  collectRuntimeScoreCandidates(*parsedOr, bestScore, foundAny);
  if (!foundAny) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "runtime profile JSON does not contain a score-like integer field: %s",
        profilePath.c_str());
  }

  return bestScore;
}

static llvm::Expected<TaskGraph>
buildCandidateGraph(const KernelArtifact &artifact,
                    const SearchInputs &inputs,
                    const CandidateExecutionSpec &candidate) {
  auto inputBindingsOr = loadInputBindings(inputs.inputFiles);
  if (!inputBindingsOr)
    return inputBindingsOr.takeError();

  auto outputBindingOr =
      buildOutputBinding(inputs.expectedFile, candidate.actualOutputPath);
  if (!outputBindingOr)
    return outputBindingOr.takeError();

  auto expectedBindingOr = buildExpectedOutputBinding(inputs.expectedFile);
  if (!expectedBindingOr)
    return expectedBindingOr.takeError();

  std::vector<uint8_t> tilingBytes =
      packTiling(candidate.params, candidate.paramTypes);
  std::optional<TilingBinding> tilingBinding;
  if (!tilingBytes.empty()) {
    tilingBinding = buildTilingBinding(tilingBytes, candidate.profileOutputDir);
    if (!tilingBinding) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "failed to materialize candidate tiling binary in %s",
          candidate.profileOutputDir.c_str());
    }
  }

  RuntimeTask task;
  task.taskId = "candidate";
  task.artifact = artifact;
  task.invocation.inputs = std::move(*inputBindingsOr);
  task.invocation.outputs.push_back(std::move(*outputBindingOr));
  task.invocation.expectedOutputs.push_back(std::move(*expectedBindingOr));
  task.invocation.tiling = std::move(tilingBinding);
  task.invocation.blockDim = static_cast<int>(candidate.blockDim);
  task.invocation.workspaceSize = 8192;
  task.invocation.enableProfiling = true;
  task.invocation.atol = inputs.atol;
  task.invocation.rtol = inputs.rtol;

  TaskGraph graph;
  if (auto err = graph.addTask(task))
    return std::move(err);
  return graph;
}

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
    } else {
      if (auto v = po->getInteger("value"))
        p.values.push_back(*v);
      if (auto* arr = po->getArray("values")) {
        for (auto& av : *arr)
          if (auto iv = av.getAsInteger()) p.values.push_back(*iv);
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
  std::string profile_path;
  std::string session_summary_path;
  std::string candidate_dir;
};

static llvm::Expected<KernelArtifact> prepareArtifact(const TilingSpace &space) {
  if (!ArtifactRoot.empty())
    return loadRuntimeSessionArtifactFromRoot(ArtifactRoot);

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
  root["kernel_file"] = space.kernel_file;

  llvm::json::Object bestObject;
  bestObject["block_dim"] = best.block_dim;
  bestObject["cycle_count"] = best.cycle_count;
  bestObject["score"] = best.cycle_count;
  bestObject["max_abs_diff"] = best.max_abs_diff;
  bestObject["profile_path"] = best.profile_path;
  bestObject["session_summary_path"] = best.session_summary_path;
  root["best"] = std::move(bestObject);

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
    const SearchInputs& searchInputs) {

  std::vector<TilingParam> search_vars;
  for (auto& p : ts.params)
    if (!p.fixed && !p.values.empty()) search_vars.push_back(p);

  std::vector<std::vector<int64_t>> combos;
  std::vector<int64_t> cur(search_vars.size(), 0);
  enumerateCombos(search_vars, 0, cur, combos);

  int total = static_cast<int>(combos.size());
  std::vector<SearchResult> results;
  ExecutionSession session(ExecutionBackendKind::Simulation);
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
        if (!p.shape_key.empty()) {
          auto it = shape.find(p.shape_key);
          if (it == shape.end()) {
            llvm::errs() << "Error: shape key '" << p.shape_key
                         << "' not found in --shape (needed by param '"
                         << p.name << "')\n";
            param_error = true;
            break;
          }
          val = it->second;
        } else if (!p.values.empty()) {
          val = p.values.front();
        } else {
          llvm::errs() << "Error: fixed param '" << p.name
                       << "' has no shape_key or value in tiling_space.json\n";
          param_error = true;
          break;
        }
      } else {
        auto it = vars.find(p.name);
        val = it != vars.end() ? it->second : 0;
      }
      param_vals.push_back({p.name, val});
      param_types.push_back(p.type);
    }
    if (param_error)
      break;

    int64_t block_dim = ts.block_dim_expr.empty() ? 1
                        : evalBlockDimExpr(ts.block_dim_expr, vars);

    llvm::outs() << "[" << (ci + 1) << "/" << total << "]";
    for (auto& kv : param_vals)
      if (kv.first.compare(0, 4, "dim_") != 0)
        llvm::outs() << " " << kv.first << "=" << kv.second;
    llvm::outs() << " block_dim=" << block_dim << " ...";
    llvm::outs().flush();

    SearchResult sr;
    sr.config       = param_vals;
    sr.block_dim    = block_dim;
    sr.cycle_count  = 0;
    sr.max_abs_diff = 0.0;

    llvm::SmallString<256> candidateDir;
    if (auto ec = llvm::sys::fs::createUniqueDirectory("autotuner_candidate", candidateDir)) {
      llvm::outs() << " FAIL  cannot create candidate directory: "
                   << ec.message() << "\n";
      llvm::outs().flush();
      results.push_back(std::move(sr));
      continue;
    }
    std::filesystem::path candidatePath(candidateDir.str().str());
    std::error_code absEc;
    candidatePath = std::filesystem::absolute(candidatePath, absEc);
    if (absEc) {
      llvm::outs() << " FAIL  cannot resolve candidate directory: "
                   << absEc.message() << "\n";
      llvm::outs().flush();
      results.push_back(std::move(sr));
      continue;
    }

    sr.candidate_dir = candidatePath.string();
    const std::filesystem::path retainedProfileRoot =
        ProfileOutDir.empty()
            ? (candidatePath / "runtime-profile")
            : std::filesystem::path(ProfileOutDir.getValue());

    CandidateExecutionSpec candidate;
    candidate.params = std::move(param_vals);
    candidate.paramTypes = std::move(param_types);
    candidate.blockDim = block_dim;
    candidate.profileOutputDir = sr.candidate_dir;
    candidate.actualOutputPath =
        (candidatePath / "output.npy").string();

    SearchInputs candidateInputs = searchInputs;
    auto graphOr = buildCandidateGraph(artifact, candidateInputs, candidate);
    if (!graphOr) {
      llvm::outs() << " FAIL  "
                   << llvm::toString(graphOr.takeError()) << "\n";
      llvm::outs().flush();
      results.push_back(std::move(sr));
      continue;
    }

    auto traceOr = session.run(*graphOr);
    if (!traceOr) {
      llvm::outs() << " FAIL  "
                   << llvm::toString(traceOr.takeError()) << "\n";
      llvm::outs().flush();
      results.push_back(std::move(sr));
      continue;
    }

    auto scoreOr = extractRuntimeScore(*traceOr);
    if (!scoreOr) {
      llvm::outs() << " FAIL  "
                   << llvm::toString(scoreOr.takeError()) << "\n";
      llvm::outs().flush();
      results.push_back(std::move(sr));
      continue;
    }

    auto retainedOr =
        retainProfileArtifactsForCli(*traceOr, retainedProfileRoot.string());
    if (!retainedOr) {
      llvm::outs() << " FAIL  "
                   << llvm::toString(retainedOr.takeError()) << "\n";
      llvm::outs().flush();
      results.push_back(std::move(sr));
      continue;
    }

    sr.cycle_count = *scoreOr;
    sr.passed = true;
    const std::vector<std::string> retainedArtifacts =
        retainedOr->profileArtifactPaths();
    if (!retainedArtifacts.empty())
      sr.profile_path = retainedArtifacts.front();
    sr.session_summary_path =
        retainedProfileSessionSummaryPath(retainedProfileRoot.string(),
                                          retainedOr->sessionId);
    llvm::outs() << " PASS score=" << sr.cycle_count << "\n";
    llvm::outs().flush();
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

  SearchInputs searchInputs;
  searchInputs.inputFiles = splitComma(InputFiles);
  searchInputs.expectedFile = ExpectedFile;
  searchInputs.shape = shape;
  searchInputs.atol = Atol;
  searchInputs.rtol = Rtol;

  llvm::outs() << "Searching " << ts.kernel_name << " on " << ts.soc << "\n";
  auto results = runSearch(ts, shape, artifact, searchInputs);

  // Filter PASS results
  std::vector<SearchResult*> passed;
  for (auto& r : results) if (r.passed) passed.push_back(&r);

  auto cleanupCandidateDirs = [&](llvm::StringRef keepDir = "") {
    for (SearchResult &result : results) {
      if (result.candidate_dir.empty() || result.candidate_dir == keepDir)
        continue;
      std::error_code ec;
      std::filesystem::remove_all(result.candidate_dir, ec);
    }
  };

  if (passed.empty()) {
    llvm::errs() << "Error: no configuration passed validation\n";
    cleanupCandidateDirs();
    _Exit(1);
  }

  // Sort by cycle_count ascending; -1 (unavailable) goes last
  std::stable_sort(passed.begin(), passed.end(), [](SearchResult* a, SearchResult* b) {
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
    cleanupCandidateDirs(best.candidate_dir);
    _Exit(1);
  }

  cleanupCandidateDirs(best.candidate_dir);

  llvm::outs().flush();
  llvm::errs().flush();
  // Use _Exit (not return/exit) to skip C++ destructors and atexit handlers:
  // runtime session teardown may leave simulator background threads active;
  // normal exit() can race those threads and segfault.
  _Exit(0);
}
