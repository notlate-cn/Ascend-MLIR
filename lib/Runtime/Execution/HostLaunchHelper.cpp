//===- HostLaunchHelper.cpp - generated-host AscendC kernel launcher --===//
//
// Implements `hostLaunchAscendCKernel`, the C-callable wrapper used by
// AclnnBackend-emitted network_host.cpp. Routes each kernel call through
// `ExecutionSession::run(TaskGraph)` (the same path runtime-session uses)
// instead of calling `NativeExecutionRunner::runFile` directly. Going
// through the session-level scheduler / SimBackend dispatch is empirically
// required for deterministic multi-block execution.
//
// Per-call I/O is staged through a temp scratch directory as .npy files so
// it matches the run-manifest contract that SimBackend already speaks.
//
//===----------------------------------------------------------------------===//

#include "Runtime/Execution/HostLaunchHelper.h"
#include "Runtime/Execution/ExecutionSession.h"
#include "Runtime/Execution/TaskGraph.h"
#include "Runtime/Support/NpyIO.h"
#include "Runtime/Support/Types.h"
#include "Runtime/TilingSchema.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir::runtime {

namespace {

// aclDataType -> runtime::DType. Returns true on success.
bool aclDtypeToDType(int aclDtype, DType &out) {
  switch (aclDtype) {
  case 0: out = DType::F32;   return true;  // ACL_FLOAT
  case 1: out = DType::F16;   return true;  // ACL_FLOAT16
  case 2: out = DType::INT8;  return true;  // ACL_INT8
  case 3: out = DType::INT32; return true;  // ACL_INT32
  case 9: out = DType::INT64; return true;  // ACL_INT64
  case 27: out = DType::BF16; return true;  // ACL_BF16
  default: return false;
  }
}

struct KernelTiling {
  int blockDim = 1;
  // Raw param values keyed by name; packed in schema-declared order at launch.
  std::unordered_map<std::string, int64_t> params;
};

class HelperState {
public:
  static HelperState &instance() {
    static HelperState inst;
    return inst;
  }

  std::mutex mu;
  std::unique_ptr<ExecutionSession> session;
  bool sessionInitialized = false;

  // Cache: parsed tilings keyed by file path.
  std::string loadedTilingsPath;
  std::unordered_map<std::string, KernelTiling> tilingsByKernel;
  bool tilingsLoaded = false;

  std::string dumpDir;
  std::string profileDir;
};

// Lazily create the ExecutionSession. Backend is Simulation by default;
// NETWORK_RUNNER_BACKEND=npu selects the real-device NpuBackend (null driver →
// runWithExecutor → NativeExecutionRunner on ASCEND_DEVICE_ID). Caller holds
// the mutex.
int ensureSession(HelperState &st) {
  if (st.sessionInitialized) return 0;
  ExecutionBackendKind kind = ExecutionBackendKind::Simulation;
  if (const char *e = std::getenv("NETWORK_RUNNER_BACKEND")) {
    if (std::string(e) == "npu")
      kind = ExecutionBackendKind::Npu;
  }
  st.session = std::make_unique<ExecutionSession>(kind);
  st.sessionInitialized = true;
  return 0;
}

// Load tilings JSON if not already loaded (or path changed). Returns 0 on
// success; missing/empty file is tolerated (no tilings is valid).
int loadTilingsIfNeeded(HelperState &st, const std::string &path) {
  if (st.tilingsLoaded && st.loadedTilingsPath == path)
    return 0;
  st.tilingsByKernel.clear();
  st.loadedTilingsPath = path;
  st.tilingsLoaded = true;
  if (path.empty()) return 0;

  auto bufOr = llvm::MemoryBuffer::getFile(path, /*IsText=*/true);
  if (!bufOr) {
    // Missing tilings file is non-fatal for kernels that need no params.
    return 0;
  }
  auto parsed = llvm::json::parse((*bufOr)->getBuffer());
  if (!parsed) {
    llvm::errs() << "hostLaunchAscendCKernel: tilings JSON parse failed ("
                 << path << "): " << llvm::toString(parsed.takeError())
                 << "\n";
    return 0;
  }
  auto *topObj = parsed->getAsObject();
  if (!topObj) return 0;

  for (auto &kv : *topObj) {
    auto *kernelObj = kv.second.getAsObject();
    if (!kernelObj) continue;
    KernelTiling t;
    for (auto &pkv : *kernelObj) {
      llvm::StringRef pname = pkv.first;
      auto valOpt = pkv.second.getAsInteger();
      if (!valOpt) continue;
      int64_t v = *valOpt;
      if (pname == "_block_dim") {
        t.blockDim = static_cast<int>(v);
        continue;
      }
      t.params[pname.str()] = v;
    }
    st.tilingsByKernel.emplace(kv.first.str(), std::move(t));
  }
  return 0;
}

// Wrap a TensorInfo into an NDArray view for SaveNpy (no allocation).
bool wrapTensorAsNDArray(const aclnn::TensorInfo &t, NDArray &out) {
  DType dt;
  if (!aclDtypeToDType(t.dtype, dt)) {
    llvm::errs() << "hostLaunchAscendCKernel: unsupported aclDataType="
                 << t.dtype << "\n";
    return false;
  }
  out.dtype = dt;
  out.shape.assign(t.shape, t.shape + t.rank);
  out.setExternal(t.data);
  return true;
}

void dumpTensorIfEnabled(const std::string &dir, const std::string &kernel,
                          const char *role, int idx,
                          const aclnn::TensorInfo &t) {
  if (dir.empty()) return;
  DType dt;
  if (!aclDtypeToDType(t.dtype, dt)) return;
  NDArray arr;
  arr.dtype = dt;
  arr.shape.assign(t.shape, t.shape + t.rank);
  arr.setExternal(t.data);
  std::string path = dir + "/" + kernel + "_" + role + "_" +
                     std::to_string(idx) + ".npy";
  if (auto err = SaveNpy(path, arr)) {
    llvm::errs() << "hostLaunchAscendCKernel: SaveNpy failed (" << path
                 << "): " << llvm::toString(std::move(err)) << "\n";
  }
}

// Make a per-call scratch directory under /tmp.
llvm::Expected<std::string> makeScratchDir() {
  char tmpl[] = "/tmp/host-launch-XXXXXX";
  if (!::mkdtemp(tmpl)) {
    return llvm::createStringError(std::error_code(errno, std::generic_category()),
                                   "mkdtemp failed");
  }
  return std::string(tmpl);
}

void removeScratchDir(const std::string &dir) {
  if (dir.empty()) return;
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  // Ignore errors — best-effort cleanup.
}

} // namespace

extern "C" void hostLaunchSetDumpIntermediatesDir(const char *dir) {
  HelperState &st = HelperState::instance();
  std::lock_guard<std::mutex> lk(st.mu);
  st.dumpDir = dir ? std::string(dir) : std::string();
}

extern "C" void hostLaunchSetProfileDir(const char *dir) {
  HelperState &st = HelperState::instance();
  std::lock_guard<std::mutex> lk(st.mu);
  st.profileDir = dir ? std::string(dir) : std::string();
}

namespace {

void writeTimingIfEnabled(const std::string &dir, const std::string &kernel,
                          int64_t wallUs, int blockDim,
                          int numInputs, int numOutputs,
                          const char *backend) {
  if (dir.empty()) return;
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  std::string path = dir + "/" + kernel + ".timing.json";
  std::FILE *fp = std::fopen(path.c_str(), "w");
  if (!fp) {
    llvm::errs() << "hostLaunchAscendCKernel: cannot write " << path << "\n";
    return;
  }
  std::fprintf(fp,
      "{\n"
      "  \"kernel\": \"%s\",\n"
      "  \"wall_us\": %lld,\n"
      "  \"block_dim\": %d,\n"
      "  \"num_inputs\": %d,\n"
      "  \"num_outputs\": %d,\n"
      "  \"backend\": \"%s\"\n"
      "}\n",
      kernel.c_str(), static_cast<long long>(wallUs), blockDim,
      numInputs, numOutputs, backend);
  std::fclose(fp);
}

} // namespace

extern "C" int hostLaunchAscendCKernel(
    const char *kernelName,
    const char *kernelBinariesDir,
    const char *tilingsPath,
    aclnn::TensorInfo *inputs, int numInputs,
    aclnn::TensorInfo *outputs, int numOutputs) {
  if (!kernelName || !kernelBinariesDir) {
    llvm::errs() << "hostLaunchAscendCKernel: kernelName/kernelBinariesDir "
                    "must not be null\n";
    return 1;
  }

  HelperState &st = HelperState::instance();
  std::lock_guard<std::mutex> lk(st.mu);

  if (int rc = ensureSession(st)) return rc;
  loadTilingsIfNeeded(st, tilingsPath ? std::string(tilingsPath)
                                       : std::string());

  // Per-call scratch dir for staged .npy files.
  auto scratchOr = makeScratchDir();
  if (!scratchOr) {
    llvm::errs() << "hostLaunchAscendCKernel(" << kernelName
                 << "): scratch dir create failed: "
                 << llvm::toString(scratchOr.takeError()) << "\n";
    return 2;
  }
  const std::string scratch = std::move(*scratchOr);

  // Build the single RuntimeTask.
  RuntimeTask task;
  task.taskId = "main";
  task.artifact.kernelName = kernelName;
  task.artifact.kernelKind = KernelKind::Vec;  // matches the magic used previously
  task.artifact.socVersion = "Ascend910B1";
  task.artifact.deviceBinaryPath = std::string(kernelBinariesDir) + "/" +
                                   kernelName + "/" + kernelName + ".bin";

  // Stage inputs as .npy files and add bindings.
  task.invocation.inputs.reserve(numInputs);
  for (int i = 0; i < numInputs; ++i) {
    NDArray view;
    if (!wrapTensorAsNDArray(inputs[i], view)) {
      removeScratchDir(scratch);
      return 2;
    }
    std::string npyPath = scratch + "/in_" + std::to_string(i) + ".npy";
    if (auto err = SaveNpy(npyPath, view)) {
      llvm::errs() << "hostLaunchAscendCKernel(" << kernelName
                   << "): SaveNpy(in_" << i << ") failed: "
                   << llvm::toString(std::move(err)) << "\n";
      removeScratchDir(scratch);
      return 2;
    }
    TensorBinding b;
    b.name = "in_" + std::to_string(i);
    b.sourceKind = BindingSourceKind::ExternalFile;
    b.path = npyPath;
    b.shape = std::vector<int64_t>(inputs[i].shape,
                                   inputs[i].shape + inputs[i].rank);
    DType dt;
    aclDtypeToDType(inputs[i].dtype, dt);
    b.dtype = dt;
    task.invocation.inputs.push_back(std::move(b));
  }

  // Add output bindings (paths the backend will write).
  task.invocation.outputs.reserve(numOutputs);
  std::vector<std::string> outputPaths;
  outputPaths.reserve(numOutputs);
  for (int i = 0; i < numOutputs; ++i) {
    DType dt;
    if (!aclDtypeToDType(outputs[i].dtype, dt)) {
      removeScratchDir(scratch);
      return 2;
    }
    std::string npyPath = scratch + "/out_" + std::to_string(i) + ".npy";
    outputPaths.push_back(npyPath);
    TensorBinding b;
    b.name = "out_" + std::to_string(i);
    b.sourceKind = BindingSourceKind::ExternalFile;
    b.path = npyPath;
    b.shape = std::vector<int64_t>(outputs[i].shape,
                                   outputs[i].shape + outputs[i].rank);
    b.dtype = dt;
    task.invocation.outputs.push_back(std::move(b));
  }

  // Tilings + block dim.
  auto tIt = st.tilingsByKernel.find(kernelName);
  if (tIt != st.tilingsByKernel.end()) {
    task.invocation.blockDim =
        tIt->second.blockDim > 0 ? tIt->second.blockDim : 1;

    std::string schemaPath = std::string(kernelBinariesDir) + "/" +
                             kernelName + "/tiling_space.json";
    auto schemaOr = TilingSchema::fromJson(schemaPath);
    if (schemaOr && schemaOr->size() > 0) {
      // Build a CSV in schema-declared order; missing values default to 0.
      std::string csv;
      for (const auto &field : schemaOr->fields()) {
        int64_t val;
        // Shape-derived (dynamic) params carry a `shape_key` ("arg<N>_dim<D>");
        // their value is the runtime extent of input N's dim D, NOT the fixed
        // (placeholder -1) entry in the tilings JSON.  Resolving it here lets
        // from-torch dynamic-shape kernels get the real row count at launch.
        int argN, dimD;
        if (!field.shapeKey.empty() &&
            std::sscanf(field.shapeKey.c_str(), "arg%d_dim%d", &argN, &dimD) ==
                2 &&
            argN >= 0 && argN < numInputs && dimD >= 0 &&
            dimD < inputs[argN].rank) {
          val = inputs[argN].shape[dimD];
        } else {
          auto pit = tIt->second.params.find(field.name);
          val = (pit != tIt->second.params.end()) ? pit->second : 0;
        }
        if (!csv.empty()) csv.push_back(',');
        csv += field.name;
        csv.push_back('=');
        csv += std::to_string(val);
      }
      TilingBinding tb;
      tb.schemaPath = schemaPath;
      tb.params = std::move(csv);
      task.invocation.tiling = std::move(tb);
    } else {
      // No schema available (missing file or empty). Pack the raw params as
      // a sequence of int64 little-endian values (matches the original
      // direct-runFile behavior) and stage as a tiling.bin binary.
      if (schemaOr) {
        // size()==0 — unusual; skip tiling.
      } else {
        llvm::consumeError(schemaOr.takeError());
      }
      if (!tIt->second.params.empty()) {
        // unordered_map iteration order is randomized per-process by libstdc++;
        // sort by key so the packed tiling bytes are deterministic across
        // runs even when no schema is available. (Without sorting, each
        // process gets a different parameter order → camodel either reads
        // garbage tiling values and silently produces zeros, or works by
        // luck. The "schema present" path above is already deterministic.)
        std::vector<std::pair<std::string, int64_t>> sortedParams(
            tIt->second.params.begin(), tIt->second.params.end());
        std::sort(sortedParams.begin(), sortedParams.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
        std::string tilingBin = scratch + "/tiling.bin";
        std::FILE *fp = std::fopen(tilingBin.c_str(), "wb");
        if (!fp) {
          llvm::errs() << "hostLaunchAscendCKernel(" << kernelName
                       << "): cannot write " << tilingBin << "\n";
          removeScratchDir(scratch);
          return 2;
        }
        for (const auto &kv : sortedParams) {
          int64_t v = kv.second;
          std::fwrite(&v, sizeof(int64_t), 1, fp);
        }
        std::fclose(fp);
        TilingBinding tb;
        tb.binaryPath = tilingBin;
        task.invocation.tiling = std::move(tb);
      }
    }
  }

  // AscendC kernels need scratch GM (workspace). Match the runtime-session
  // manifest convention (16 MiB). Override via NETWORK_RUNNER_WORKSPACE_BYTES.
  task.invocation.workspaceSize = 16 * 1024 * 1024;
  if (const char *e = std::getenv("NETWORK_RUNNER_WORKSPACE_BYTES"))
    task.invocation.workspaceSize = std::strtoull(e, nullptr, 10);

  // Dump inputs (caller-owned buffers, before launch).
  for (int i = 0; i < numInputs; ++i)
    dumpTensorIfEnabled(st.dumpDir, kernelName, "in", i, inputs[i]);

  // Build TaskGraph and run via ExecutionSession (same path as runtime-session).
  TaskGraph graph;
  if (auto err = graph.addTask(task)) {
    llvm::errs() << "hostLaunchAscendCKernel(" << kernelName
                 << "): graph.addTask failed: "
                 << llvm::toString(std::move(err)) << "\n";
    removeScratchDir(scratch);
    return 3;
  }

  // Wall-clock around session->run for per-kernel timing. Caveats: on sim
  // this is camodel exec time; on NPU it includes dispatch overhead.
  auto t0 = std::chrono::steady_clock::now();
  auto traceOr = st.session->run(graph);
  auto t1 = std::chrono::steady_clock::now();
  int64_t wallUs = std::chrono::duration_cast<std::chrono::microseconds>(
      t1 - t0).count();
  if (!traceOr) {
    llvm::errs() << "hostLaunchAscendCKernel(" << kernelName
                 << ") session.run failed: "
                 << llvm::toString(traceOr.takeError()) << "\n";
    removeScratchDir(scratch);
    return 3;
  }
  {
    const char *be = "sim";
    if (const char *e = std::getenv("NETWORK_RUNNER_BACKEND"))
      if (std::string(e) == "npu") be = "npu";
    writeTimingIfEnabled(st.profileDir, kernelName, wallUs,
                         task.invocation.blockDim, numInputs, numOutputs, be);
  }

  // Load outputs back from .npy and memcpy into caller buffers.
  for (int i = 0; i < numOutputs; ++i) {
    auto arrOr = LoadNpy(outputPaths[i]);
    if (!arrOr) {
      llvm::errs() << "hostLaunchAscendCKernel(" << kernelName
                   << "): LoadNpy(out_" << i << ") failed: "
                   << llvm::toString(arrOr.takeError()) << "\n";
      removeScratchDir(scratch);
      return 3;
    }
    NDArray &arr = *arrOr;
    DType expectDt;
    aclDtypeToDType(outputs[i].dtype, expectDt);
    size_t expectBytes = arr.nbytes();
    // Sanity: make sure it matches caller buffer size.
    NDArray expectView;
    expectView.dtype = expectDt;
    expectView.shape.assign(outputs[i].shape,
                            outputs[i].shape + outputs[i].rank);
    if (expectView.nbytes() != expectBytes) {
      llvm::errs() << "hostLaunchAscendCKernel(" << kernelName
                   << "): output " << i << " size mismatch (expected "
                   << expectView.nbytes() << " got " << expectBytes << ")\n";
      removeScratchDir(scratch);
      return 3;
    }
    std::memcpy(outputs[i].data, arr.data, expectBytes);
  }

  // Dump outputs (post-launch caller buffers).
  for (int i = 0; i < numOutputs; ++i)
    dumpTensorIfEnabled(st.dumpDir, kernelName, "out", i, outputs[i]);

  removeScratchDir(scratch);
  return 0;
}

} // namespace mlir::runtime
