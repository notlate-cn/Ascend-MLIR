// tools/autotuner/autotuner_main.cpp
#include "Runtime/Compiler.h"
#include "Runtime/Executor.h"
#include "Runtime/NpyIO.h"
#include "Runtime/SimValidator.h"
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
    cl::desc("Output tiling_func.cpp path"), cl::init("tiling_func.cpp"));
static cl::opt<std::string> SocVersion("soc",
    cl::desc("SoC version (overrides JSON; default: Ascend910B1)"), cl::init(""));
static cl::opt<double> Atol("atol", cl::desc("Absolute tolerance"), cl::init(1.0));
static cl::opt<double> Rtol("rtol", cl::desc("Relative tolerance"), cl::init(1e-2));

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
  std::string soc;
  std::string block_dim_expr;
  std::vector<TilingParam> params;
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
    const std::string& kernel_path,
    const std::string& soc,
    RunArgs& args_template,
    const std::vector<NDArray>& expected,
    double atol, double rtol) {

  std::vector<TilingParam> search_vars;
  for (auto& p : ts.params)
    if (!p.fixed && !p.values.empty()) search_vars.push_back(p);

  std::vector<std::vector<int64_t>> combos;
  std::vector<int64_t> cur(search_vars.size(), 0);
  enumerateCombos(search_vars, 0, cur, combos);

  int total = static_cast<int>(combos.size());
  std::vector<SearchResult> results;
  Compiler::Config cc;
  cc.soc_version = soc;

  // Compile kernel once — all configs share the same ELF binary (tiling is
  // passed as kernel arguments, not compiled-in). Reusing the binary avoids
  // re-registering the same binary in the simulator's internal registry, which
  // causes rtFunctionRegister rc=507000 on the 2nd+ registration.
  llvm::SmallString<256> build_dir;
  if (llvm::sys::fs::createUniqueDirectory("autotuner_build", build_dir)) {
    llvm::errs() << "Error: cannot create temp build dir\n";
    return results;
  }
  Compiler compiler(cc);
  auto bin_or = compiler.Compile(kernel_path, build_dir.str().str(), ts.kernel_name);
  if (!bin_or) {
    llvm::errs() << "Error: compile failed: " << llvm::toString(bin_or.takeError()) << "\n";
    return results;
  }
  std::string binary_path = *bin_or;

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
  auto handle_or = executor.RegisterBinary(binary_path, ts.kernel_name);
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

    RunArgs args = args_template;
    args.tiling    = packTiling(param_vals, param_types);
    args.block_dim = static_cast<int>(block_dim);

    auto res = validator.ValidateBinary(func_handle, executor, args,
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

// ── Code generator ────────────────────────────────────────────────────────────

static void emitTilingFunc(
    const std::string& out_path,
    const TilingSpace& ts,
    const SearchResult& best,
    const std::map<std::string, int64_t>& /*shape*/) {

  // Collect unique shape params (from fixed param shape_keys, in order)
  std::vector<std::string> shape_params;
  for (auto& p : ts.params)
    if (p.fixed && !p.shape_key.empty()) {
      bool found = false;
      for (auto& s : shape_params) if (s == p.shape_key) { found = true; break; }
      if (!found) shape_params.push_back(p.shape_key);
    }

  std::ofstream f(out_path);
  if (!f) {
    llvm::errs() << "Error: cannot write to " << out_path << "\n";
    _Exit(1);
  }

  f << "// Auto-generated by autotuner\n";
  f << "// kernel: " << ts.kernel_name << "  soc: " << ts.soc << "\n";
  f << "// best config:";
  for (auto& kv : best.config)
    if (kv.first.compare(0, 4, "dim_") != 0) f << "  " << kv.first << "=" << kv.second;
  f << "\n";
  f << "// cycle_count=" << best.cycle_count
    << "  max_abs_diff=" << best.max_abs_diff << "\n\n";

  f << "#include <cstdint>\n\n";

  // TilingData struct
  f << "struct TilingData {\n";
  for (auto& p : ts.params)
    f << "  " << (p.type == "int32" ? "int32_t" : "int64_t")
      << " " << p.name << ";\n";
  f << "};\n\n";

  // Shape param signature
  std::string shape_sig;
  for (size_t i = 0; i < shape_params.size(); ++i) {
    if (i) shape_sig += ", ";
    shape_sig += "int64_t " + shape_params[i];
  }

  // get_tiling()
  f << "void get_tiling(" << shape_sig << ", TilingData* out) {\n";
  for (auto& kv : best.config) {
    bool is_fixed = false;
    std::string sk;
    for (auto& p : ts.params)
      if (p.name == kv.first && p.fixed) { is_fixed = true; sk = p.shape_key; break; }
    if (is_fixed)
      f << "  out->" << kv.first << " = " << sk << ";\n";
    else
      f << "  out->" << kv.first << " = " << kv.second << ";\n";
  }
  f << "}\n\n";

  // get_block_dim()
  // Substitute search-var constants into block_dim_expr, then rewrite
  // "ceil(A/B)" -> "(A + B - 1) / B" for correct integer ceiling.
  // Sort substitution names longest-first to avoid partial matches
  // (e.g. "TB_M" must be substituted before "M").
  f << "int get_block_dim(" << shape_sig << ") {\n";
  if (!ts.block_dim_expr.empty()) {
    std::string expr = ts.block_dim_expr;
    expr.erase(std::remove(expr.begin(), expr.end(), ' '), expr.end());

    std::vector<std::pair<std::string, int64_t>> subst_list;
    for (auto& kv : best.config) {
      bool is_fixed = false;
      for (auto& p : ts.params) if (p.name == kv.first && p.fixed) { is_fixed = true; break; }
      if (!is_fixed) subst_list.push_back({kv.first, kv.second});
    }
    std::sort(subst_list.begin(), subst_list.end(),
              [](const std::pair<std::string,int64_t>& a,
                 const std::pair<std::string,int64_t>& b){
                return a.first.size() > b.first.size();
              });

    for (auto& sv : subst_list) {
      const std::string& from = sv.first;
      std::string to = std::to_string(sv.second);
      size_t pos = 0;
      while ((pos = expr.find(from, pos)) != std::string::npos) {
        bool pre_ok  = (pos == 0) || (!std::isalnum((unsigned char)expr[pos-1]) && expr[pos-1] != '_');
        bool post_ok = (pos + from.size() >= expr.size()) ||
                       (!std::isalnum((unsigned char)expr[pos+from.size()]) && expr[pos+from.size()] != '_');
        if (pre_ok && post_ok) {
          expr.replace(pos, from.size(), to);
          pos += to.size();
        } else {
          pos += from.size();
        }
      }
    }

    // Rewrite "ceil(A/B)" -> "(A + B - 1) / B"
    std::string emit_expr = expr;
    if (expr.size() > 5 && expr.substr(0, 5) == "ceil(" && expr.back() == ')') {
      std::string inner = expr.substr(5, expr.size() - 6);
      auto slash = inner.find('/');
      if (slash != std::string::npos) {
        std::string lhs = inner.substr(0, slash);
        std::string rhs = inner.substr(slash + 1);
        emit_expr = "(" + lhs + " + " + rhs + " - 1) / " + rhs;
      }
    }

    f << "  // block_dim_expr: " << ts.block_dim_expr << "\n";
    f << "  return static_cast<int>(" << emit_expr << ");\n";
  } else {
    f << "  return " << best.block_dim << ";\n";
  }
  f << "}\n";

  llvm::outs() << "Emitted: " << out_path << "\n";
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

  // --kernel overrides JSON kernel_file; relative paths are resolved differently:
  //   kernel_file (from JSON) → relative to the JSON file's directory
  //   --kernel (from CLI)     → relative to cwd (standard shell convention)
  bool kernel_from_cli = !KernelFile.empty();
  std::string kernel_path = kernel_from_cli ? KernelFile.getValue() : ts.kernel_file;
  if (kernel_path.empty()) {
    llvm::errs() << "Error: kernel file not specified (--kernel or kernel_file in JSON)\n";
    _Exit(1);
  }
  if (!llvm::sys::path::is_absolute(kernel_path)) {
    if (!kernel_from_cli) {
      // kernel_file in JSON: resolve relative to the JSON file's directory
      llvm::SmallString<256> base(SpaceFile.getValue());
      llvm::sys::path::remove_filename(base);
      llvm::sys::path::append(base, kernel_path);
      kernel_path = base.str().str();
    }
    // else: --kernel is relative to cwd — leave as-is (Compiler resolves via cwd)
  }

  std::string soc = SocVersion.empty() ? ts.soc : SocVersion.getValue();
  if (soc.empty()) soc = "Ascend910B1";
  ts.soc = soc;

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
    inputs.push_back(*arr_or);
  }

  // Load expected
  auto exp_or = LoadNpy(ExpectedFile);
  if (!exp_or) {
    for (auto& inp : inputs) delete[] static_cast<uint8_t*>(inp.data);
    llvm::errs() << "Error loading expected: " << llvm::toString(exp_or.takeError()) << "\n";
    _Exit(1);
  }
  std::vector<NDArray> expected_arrs = {*exp_or};

  // Pre-alloc output buffer (shape/dtype from expected)
  NDArray out_buf;
  out_buf.shape = expected_arrs[0].shape;
  out_buf.dtype = expected_arrs[0].dtype;
  out_buf.data  = new uint8_t[out_buf.nbytes()]();

  RunArgs args_tmpl;
  args_tmpl.inputs  = inputs;
  args_tmpl.outputs = {out_buf};

  llvm::outs() << "Searching " << ts.kernel_name << " on " << soc << "\n";
  auto results = runSearch(ts, shape, kernel_path, soc,
                            args_tmpl, expected_arrs, Atol, Rtol);

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

  emitTilingFunc(OutputFile, ts, best, shape);

  llvm::outs().flush();
  llvm::errs().flush();
  // Use _Exit (not return/exit) to skip C++ destructors and atexit handlers:
  // SimValidator invokes libruntime_camodel which leaves background simulator
  // threads running; normal exit() races those threads and segfaults.
  _Exit(0);
}
