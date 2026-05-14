#include "Runtime/AclnnBackend.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <string>

namespace mlir::runtime {

namespace {

// ---------------------------------------------------------------------------
// CoordEmitter: walks the coordinator func and emits the network_impl body.
// ---------------------------------------------------------------------------
class CoordEmitter {
public:
  CoordEmitter(ModuleOp module, llvm::raw_ostream &os,
               const AclnnBackendConfig &cfg)
      : module_(module), os_(os), cfg_(cfg) {}

  void emit(func::FuncOp coord) {
    // Map func arguments to inputs[N].
    for (auto [i, arg] : llvm::enumerate(coord.getArguments()))
      names_[arg] = "inputs[" + std::to_string(i) + "]";

    for (Operation &op : coord.front())
      emitOp(op);
  }

private:
  std::string fresh() { return "t" + std::to_string(nextTmp_++); }

  std::string nameOf(Value v) {
    auto it = names_.find(v);
    if (it != names_.end()) return it->second;
    return "/*unknown*/";
  }

  void emitOp(Operation &op) {
    if (auto callOp = dyn_cast<func::CallOp>(op))
      return emitCall(callOp);
    if (auto castOp = dyn_cast<tensor::CastOp>(op)) {
      names_[castOp.getResult()] = nameOf(castOp.getSource());
      return;
    }
    if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
      names_[collapseOp.getResult()] = nameOf(collapseOp.getSrc());
      return;
    }
    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
      names_[expandOp.getResult()] = nameOf(expandOp.getSrc());
      return;
    }
    if (auto emptyOp = dyn_cast<tensor::EmptyOp>(op)) {
      std::string n = fresh();
      os_ << "  TensorInfo " << n << "; // tensor.empty — allocate at runtime\n";
      names_[emptyOp.getResult()] = n;
      return;
    }
    if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
      emitReturn(retOp);
      return;
    }
    // arith constants, linalg.fill, etc. — not needed for pure aclnn path.
  }

  void emitCall(func::CallOp callOp) {
    auto callee = module_.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (!callee) {
      os_ << "  // WARNING: unknown callee " << callOp.getCallee() << "\n";
      return;
    }

    if (auto opAttr = callee->getAttrOfType<StringAttr>("aclnn.op")) {
      // aclnn direct call → emit run_<Op>(...).
      // All results share one TensorInfo (aclnn ops return a single tensor).
      std::string resName = fresh();
      os_ << "  TensorInfo " << resName << ";\n";
      os_ << "  run_" << opAttr.getValue() << "(";
      for (auto [i, arg] : llvm::enumerate(callOp.getOperands())) {
        if (i) os_ << ", ";
        os_ << nameOf(arg);
      }
      os_ << ", &" << resName << ", stream);\n";
      for (auto res : callOp.getResults())
        names_[res] = resName;
      return;
    }

    // AscendC kernel group → call the host launch helper.
    emitAscendCLaunch(callOp);
  }

  // Map MLIR element type → aclDataType numeric id used by TensorInfo.dtype.
  // f16 -> 1 (ACL_FLOAT16), bf16 -> 27, f32 -> 0.  Returns -1 if unknown.
  static int dtypeIdFor(Type elemTy) {
    if (elemTy.isF16()) return 1;
    if (elemTy.isBF16()) return 27;
    if (elemTy.isF32()) return 0;
    return -1;
  }

  // Element size in bytes for the supported dtypes.
  static int elemBytesFor(Type elemTy) {
    if (elemTy.isF16() || elemTy.isBF16()) return 2;
    if (elemTy.isF32()) return 4;
    return 0;
  }

  void emitAscendCLaunch(func::CallOp callOp) {
    const auto kernelName = callOp.getCallee().str();
    const int numIn = static_cast<int>(callOp.getNumOperands());
    const int numOut = static_cast<int>(callOp.getNumResults());

    std::string insName = fresh();
    std::string outsName = fresh();

    // Build input TensorInfo array first; outputs may need its shapes for
    // dynamic dims.
    os_ << "  TensorInfo " << insName << "[" << std::max(numIn, 1) << "] = {";
    for (auto [i, arg] : llvm::enumerate(callOp.getOperands())) {
      if (i) os_ << ", ";
      os_ << nameOf(arg);
    }
    os_ << "};\n";

    // Preallocate output TensorInfos. For dynamic dims (kDynamic in the IR),
    // copy the dim from the kernel's first input at runtime — works for the v1
    // elementwise kernels where output shape == first-input shape.
    os_ << "  TensorInfo " << outsName << "[" << std::max(numOut, 1) << "] = {};\n";
    for (auto [ri, res] : llvm::enumerate(callOp.getResults())) {
      auto rtt = dyn_cast<RankedTensorType>(res.getType());
      if (!rtt) {
        os_ << "  // WARNING: non-ranked-tensor result for " << kernelName << "\n";
        continue;
      }
      auto shape = rtt.getShape();
      int dtypeId = dtypeIdFor(rtt.getElementType());
      int elemBytes = elemBytesFor(rtt.getElementType());
      if (dtypeId == -1 || elemBytes == 0) {
        module_.emitError()
            << "AclnnBackend: unsupported element type '"
            << rtt.getElementType() << "' for result " << ri << " of '"
            << kernelName << "'; cannot emit TensorInfo";
        os_ << "#error \"unsupported element type for " << kernelName
            << " result " << ri << "\"\n";
        continue;
      }
      os_ << "  " << outsName << "[" << ri << "].rank = " << shape.size() << ";\n";
      for (auto [di, d] : llvm::enumerate(shape)) {
        if (mlir::ShapedType::isDynamic(d)) {
          os_ << "  " << outsName << "[" << ri << "].shape[" << di << "] = "
              << insName << "[0].shape[" << di << "];\n";
        } else {
          os_ << "  " << outsName << "[" << ri << "].shape[" << di << "] = "
              << d << ";\n";
        }
      }
      os_ << "  " << outsName << "[" << ri << "].dtype = " << dtypeId << ";\n";
      // Compute byte size at runtime so dynamic dims work.
      os_ << "  { size_t _n = " << elemBytes << "; for (int _d = 0; _d < "
          << outsName << "[" << ri << "].rank; ++_d) _n *= (size_t)"
          << outsName << "[" << ri << "].shape[_d]; "
          << outsName << "[" << ri << "].data = ::operator new(_n); }\n";
    }

    // Call the helper.
    os_ << "  if (mlir::runtime::hostLaunchAscendCKernel(\n";
    os_ << "        \"" << kernelName << "\",\n";
    os_ << "        /*kernelBinariesDir=*/\"" << cfg_.kernelBinariesDir
        << "\",\n";
    os_ << "        /*tilingsPath=*/\"" << cfg_.tilingsPath << "\",\n";
    os_ << "        " << insName << ", " << numIn << ", "
        << outsName << ", " << numOut << ") != 0) {\n";
    os_ << "    fprintf(stderr, \"hostLaunchAscendCKernel(" << kernelName
        << ") failed\\n\");\n";
    os_ << "    return;\n";
    os_ << "  }\n";

    // Register result names.
    for (auto [ri, res] : llvm::enumerate(callOp.getResults()))
      names_[res] = outsName + "[" + std::to_string(ri) + "]";
  }

  void emitReturn(func::ReturnOp retOp) {
    for (auto [i, v] : llvm::enumerate(retOp.getOperands()))
      os_ << "  outputs[" << i << "] = " << nameOf(v) << ";\n";
  }

  ModuleOp module_;
  llvm::raw_ostream &os_;
  const AclnnBackendConfig &cfg_;
  llvm::DenseMap<Value, std::string> names_;
  int nextTmp_ = 0;
};

// ---------------------------------------------------------------------------
// Identify the coordinator function: the non-private func in the module.
// ---------------------------------------------------------------------------
static func::FuncOp findCoordinator(ModuleOp module) {
  func::FuncOp coord;
  module.walk([&](func::FuncOp f) {
    if (!f.isPrivate()) {
      coord = f;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return coord;
}

// ---------------------------------------------------------------------------
// Collect unique aclnn.op values referenced in the coordinator.
// ---------------------------------------------------------------------------
static void collectAclnnOps(ModuleOp module, func::FuncOp coord,
                             llvm::SmallVector<std::string> &ops) {
  llvm::StringSet<> seen;
  for (Operation &op : coord.front()) {
    auto callOp = dyn_cast<func::CallOp>(op);
    if (!callOp) continue;
    auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (!callee) continue;
    auto opAttr = callee->getAttrOfType<StringAttr>("aclnn.op");
    if (!opAttr) continue;
    auto name = opAttr.getValue().str();
    if (seen.insert(name).second)
      ops.push_back(name);
  }
}

// ---------------------------------------------------------------------------
// Build the complete network_host.cpp source string.
// ---------------------------------------------------------------------------
static std::string buildNetworkHostCpp(ModuleOp module,
                                       const AclnnBackendConfig &cfg) {
  std::string src;
  llvm::raw_string_ostream os(src);

  func::FuncOp coord = findCoordinator(module);
  if (!coord) {
    os << "// ERROR: no coordinator function found in network.mlir\n";
    return src;
  }

  llvm::SmallVector<std::string> aclnnOps;
  collectAclnnOps(module, coord, aclnnOps);

  // ① File header + includes
  os << "// Auto-generated by AclnnBackend. DO NOT EDIT.\n";
  if (!cfg.tilingsPath.empty())
    os << "// tilings_path: " << cfg.tilingsPath << "\n";
  if (!cfg.kernelBinariesDir.empty())
    os << "// kernel_binaries_dir: " << cfg.kernelBinariesDir << "\n";
  os << "#include \"acl/acl.h\"\n";
  os << "#include \"Runtime/AclnnOps.h\"\n";
  os << "#include \"Runtime/Execution/HostLaunchHelper.h\"\n";
  os << "#include <cstdint>\n";
  os << "#include <cstdio>\n";
  os << "#include <cstring>\n";
  os << "#include <cstdlib>\n";
  os << "\n";
  os << "using TensorInfo = mlir::runtime::aclnn::TensorInfo;\n";
  os << "using mlir::runtime::aclnn::run_" << (!aclnnOps.empty() ? aclnnOps[0] : "FlashAttentionScore") << ";\n";
  os << "\n";

  // ③ network_impl: coordinator body translated to C++
  (void)coord.getNumArguments();      // counted by CoordEmitter via block args
  (void)coord.getResultTypes().size();
  os << "static void network_impl(\n";
  os << "    TensorInfo inputs[], int /*numInputs*/,\n";
  os << "    TensorInfo outputs[], int /*numOutputs*/,\n";
  os << "    aclrtStream stream) {\n";

  {
    CoordEmitter emitter(module, os, cfg);
    emitter.emit(coord);
  }

  os << "}\n\n";

  // ④ Public entry point — extern "C" so harness.cpp can declare it without mangling
  os << "extern \"C\" void network(\n";
  os << "    TensorInfo inputs[], int numInputs,\n";
  os << "    TensorInfo outputs[], int numOutputs,\n";
  os << "    aclrtStream stream) {\n";
  os << "  // Try aclInit; if it fails (e.g. running on CPU sim with no NPU\n";
  os << "  // device available), fall through in host-mode so aclnn ops dispatch\n";
  os << "  // to AclnnOps.cpp's CPU-reference implementations.\n";
  os << "  static bool initialized = false;\n";
  os << "  if (!initialized) {\n";
  os << "    initialized = true;\n";
  os << "    const char *force = std::getenv(\"ASCEND_MLIR_FORCE_HOST_MODE\");\n";
  os << "    if (force && force[0] && force[0] != '0') {\n";
  os << "      mlir::runtime::aclnn::setHostMode(true);\n";
  os << "    } else {\n";
  os << "      int rc = aclInit(nullptr);\n";
  os << "      if (rc != ACL_SUCCESS && rc != ACL_ERROR_REPEAT_INITIALIZE) {\n";
  os << "        mlir::runtime::aclnn::setHostMode(true);\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "  network_impl(inputs, numInputs, outputs, numOutputs, stream);\n";
  os << "}\n";

  os << "\nextern \"C\" void network_set_dump_dir(const char *dir) {\n";
  os << "  mlir::runtime::hostLaunchSetDumpIntermediatesDir(dir);\n";
  os << "}\n";

  return src;
}

} // namespace

// ---------------------------------------------------------------------------
// AclnnBackend::generate
// ---------------------------------------------------------------------------
llvm::Error AclnnBackend::generate(const AclnnBackendConfig &cfg) {
  MLIRContext ctx;
  ctx.loadDialect<func::FuncDialect,
                  tensor::TensorDialect,
                  arith::ArithDialect,
                  linalg::LinalgDialect>();

  auto moduleRef = parseSourceFile<ModuleOp>(cfg.networkMlirPath, &ctx);
  if (!moduleRef)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "AclnnBackend: failed to parse %s",
                                   cfg.networkMlirPath.c_str());

  std::string cppSrc = buildNetworkHostCpp(*moduleRef, cfg);

  std::ofstream out(cfg.outputCppPath, std::ios::binary);
  if (!out)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "AclnnBackend: cannot write %s",
                                   cfg.outputCppPath.c_str());
  out << cppSrc;
  if (!out)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "AclnnBackend: write failed for %s",
                                   cfg.outputCppPath.c_str());

  return llvm::Error::success();
}

} // namespace mlir::runtime
