# AutoTunerPass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an MLIR pass (`--autotuner`) that reads step7_kernel.mlir, extracts TilingData schema from `emitasc.py_struct`, searches for optimal tiling parameters via enumeration, and emits Host C++ `tiling_func_i.cpp` + `get_tiling.cpp`.

**Architecture:** Five cooperating components (HardwareProfile → TilingSpace → MlirAnalyzer → EnumerateSolver → TilingFuncEmitter) assembled by AutoTunerPass. No existing passes are modified. Follows the same TableGen + CMake pattern as other passes in `lib/Conversion/`. Prerequisite: SimValidator plan must be complete (lib/Runtime/ must exist and build).

**Tech Stack:** MLIR C++ (ModuleOp, FuncOp, AffineExpr, mlir::presburger::IntegerRelation), LLVM Support (Expected/Error, JSON), C++17.

---

## File Map

| File | Responsibility |
|------|---------------|
| `include/AutoTuner/HardwareProfile.h` | SoC hardware params struct + JSON load |
| `include/AutoTuner/TilingSpace.h` | Search space + dual-path constraint system |
| `include/AutoTuner/MlirAnalyzer.h` | KernelAnalysis, TilingField, PipeAccess structs + Analyze() |
| `include/AutoTuner/Solver.h` | TilingExprResult, SolverBase, EnumerateSolver |
| `include/AutoTuner/TilingFuncEmitter.h` | Emitter class declaration |
| `include/AutoTuner/AutoTunerPass.h` | Pass factory declaration |
| `lib/AutoTuner/HardwareProfile.cpp` | JSON parsing via llvm::json |
| `lib/AutoTuner/TilingSpace.cpp` | JSON load, constraint parsing, IntegerRelation build |
| `lib/AutoTuner/MlirAnalyzer.cpp` | Walk step7 IR, extract tiling_fields/shape_dims/pipe_access |
| `lib/AutoTuner/Solver.cpp` | EnumerateSolver: enumerate + prune + score |
| `lib/AutoTuner/TilingFuncEmitter.cpp` | AffineExpr → C++ text codegen |
| `lib/AutoTuner/AutoTunerPass.cpp` | Pass registration, orchestration |
| `lib/AutoTuner/CMakeLists.txt` | Build rules |
| `include/Conversion/Passes.td` | Add AutoTunerPass TableGen entry |
| `include/Conversion/Passes.h` | Add AutoTunerPass.h include |
| `lib/CMakeLists.txt` | Add subdirectory(AutoTuner) |
| `tools/afir-opt/CMakeLists.txt` | Link AutoTunerConversion |
| `hardware/Ascend910B1.json` | Hardware profile JSON |
| `hardware/Ascend910B2.json` | Hardware profile JSON (stub) |
| `test/AutoTuner/autotuner_smoke.mlir` | Lit test: run pass on broadcast-add-reduce step7 |

---

## Task 1: HardwareProfile

**Files:**
- Create: `include/AutoTuner/HardwareProfile.h`
- Create: `lib/AutoTuner/HardwareProfile.cpp`
- Create: `hardware/Ascend910B1.json`
- Create: `hardware/Ascend910B2.json`

- [ ] **Step 1: Write `include/AutoTuner/HardwareProfile.h`**

```cpp
// include/AutoTuner/HardwareProfile.h
#pragma once
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>

namespace mlir::autotuner {

struct HardwareProfile {
  std::string soc_name;
  int64_t     ub_size         = 204800;  // bytes
  int32_t     aiv_num         = 20;      // AIV core count (compile-time constant)
  double      mte2_bandwidth  = 800.0;   // GB/s  GM→UB
  double      mte3_bandwidth  = 800.0;   // GB/s  UB→GM

  // soc_name → {hw_dir}/{soc_name}.json
  // hw_dir: defaults to CMAKE_INSTALL_PREFIX/share/autotuner/hardware/ if empty
  static llvm::Expected<HardwareProfile> Load(const std::string& soc_name,
                                              const std::string& hw_dir = "");
};

} // namespace mlir::autotuner
```

- [ ] **Step 2: Write `lib/AutoTuner/HardwareProfile.cpp`**

```cpp
// lib/AutoTuner/HardwareProfile.cpp
#include "AutoTuner/HardwareProfile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

namespace mlir::autotuner {

llvm::Expected<HardwareProfile> HardwareProfile::Load(
    const std::string& soc_name, const std::string& hw_dir) {

  // Resolve hw_dir: if empty, use AUTOTUNER_HW_DIR compile-time default
  std::string dir = hw_dir;
  if (dir.empty()) {
#ifndef AUTOTUNER_HW_DIR
#define AUTOTUNER_HW_DIR ""
#endif
    dir = AUTOTUNER_HW_DIR;
  }
  if (dir.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "hw_dir not specified and AUTOTUNER_HW_DIR not set");

  llvm::SmallString<256> json_path(dir);
  llvm::sys::path::append(json_path, soc_name + ".json");

  auto buf = llvm::MemoryBuffer::getFile(json_path);
  if (!buf)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot read %s", json_path.c_str());

  auto json = llvm::json::parse((*buf)->getBuffer());
  if (!json)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "JSON parse error in %s", json_path.c_str());

  const llvm::json::Object* obj = json->getAsObject();
  if (!obj)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Expected JSON object in %s", json_path.c_str());

  HardwareProfile hw;
  hw.soc_name = soc_name;
  if (auto v = obj->getInteger("ub_size"))       hw.ub_size        = *v;
  if (auto v = obj->getInteger("aiv_num"))       hw.aiv_num        = static_cast<int32_t>(*v);
  if (auto v = obj->getNumber("mte2_bandwidth")) hw.mte2_bandwidth = *v;
  if (auto v = obj->getNumber("mte3_bandwidth")) hw.mte3_bandwidth = *v;
  return hw;
}

} // namespace mlir::autotuner
```

- [ ] **Step 3: Create hardware JSON files**

`hardware/Ascend910B1.json`:
```json
{
  "soc_name":        "Ascend910B1",
  "ub_size":         204800,
  "aiv_num":         20,
  "mte2_bandwidth":  800.0,
  "mte3_bandwidth":  800.0
}
```

`hardware/Ascend910B2.json`:
```json
{
  "soc_name":        "Ascend910B2",
  "ub_size":         262144,
  "aiv_num":         20,
  "mte2_bandwidth":  800.0,
  "mte3_bandwidth":  800.0
}
```

- [ ] **Step 4: Create stub `lib/AutoTuner/CMakeLists.txt`**

```cmake
# lib/AutoTuner/CMakeLists.txt
add_mlir_library(AutoTunerConversion
  HardwareProfile.cpp
  TilingSpace.cpp
  MlirAnalyzer.cpp
  Solver.cpp
  TilingFuncEmitter.cpp
  AutoTunerPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/AutoTuner

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRAffine
  LLVMSupport
  AFIRDialect
)
```

Create empty placeholder files for later tasks:
```bash
touch lib/AutoTuner/TilingSpace.cpp
touch lib/AutoTuner/MlirAnalyzer.cpp
touch lib/AutoTuner/Solver.cpp
touch lib/AutoTuner/TilingFuncEmitter.cpp
touch lib/AutoTuner/AutoTunerPass.cpp
```

- [ ] **Step 5: Commit**

```bash
git add include/AutoTuner/HardwareProfile.h lib/AutoTuner/HardwareProfile.cpp \
        lib/AutoTuner/CMakeLists.txt \
        lib/AutoTuner/TilingSpace.cpp lib/AutoTuner/MlirAnalyzer.cpp \
        lib/AutoTuner/Solver.cpp lib/AutoTuner/TilingFuncEmitter.cpp \
        lib/AutoTuner/AutoTunerPass.cpp \
        hardware/Ascend910B1.json hardware/Ascend910B2.json
git commit -m "feat(autotuner): add HardwareProfile + hardware JSON"
```

---

## Task 2: TilingSpace

**Files:**
- Create: `include/AutoTuner/TilingSpace.h`
- Modify: `lib/AutoTuner/TilingSpace.cpp`

Background: `tiling_space.json` describes per-param ranges (`min`, `max`, `step`, `alignment`) and constraint strings. Constraints are parsed into two buckets:
- **Linear** (e.g., `TB_N <= N`) → `IntegerRelation`
- **Nonlinear** (e.g., `TB_M * TB_N * 2 <= ub_size / 2`) → stored as strings, evaluated pointwise during enumeration

`mlir::presburger::IntegerRelation` (from `mlir/Analysis/Presburger/IntegerRelation.h`) represents a system of linear constraints over integer variables. Variables are indexed 0..n-1; to add constraint `TB_M <= M`, call `addInequality({1, -1, 0})` meaning `TB_M - M <= 0`.

- [ ] **Step 1: Write `include/AutoTuner/TilingSpace.h`**

```cpp
// include/AutoTuner/TilingSpace.h
#pragma once
#include "AutoTuner/HardwareProfile.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mlir::autotuner {

struct TilingParam {
  std::string name;
  int64_t     min, max, step, alignment;
};

struct TilingSpace {
  std::vector<TilingParam> params;

  // Linear constraints as IntegerRelation.
  // Variable order: [params[0], params[1], ..., shape_vars..., hw_consts...]
  // Shape vars (M, N, ...) are treated as free variables when building from JSON;
  // during enumeration, caller substitutes concrete shape values.
  mlir::presburger::IntegerRelation linear_constraints;

  // Nonlinear constraint strings (evaluated pointwise in Solver).
  // Format: same as JSON "constraints" strings.
  std::vector<std::string> nonlinear_exprs;

  // Evaluate a single nonlinear constraint string with given variable bindings.
  // Returns true if constraint is satisfied.
  // bindings: map from variable name → value (tiling params + shape dims + hw consts)
  static bool EvalConstraint(
      const std::string& expr,
      const std::vector<std::pair<std::string, int64_t>>& bindings);

  // Load from JSON file.
  // Priority: {search_dir}/{func_name}_tiling_space.json
  //         > {search_dir}/tiling_space.json
  //         > explicit_path (if non-empty)
  // hw: used to substitute ub_size/aiv_num as constants in constraints.
  static llvm::Expected<TilingSpace> LoadFromJson(
      const std::string& func_name,
      const std::string& search_dir,
      const HardwareProfile& hw,
      const std::string& explicit_path = "");
};

} // namespace mlir::autotuner
```

- [ ] **Step 2: Write `lib/AutoTuner/TilingSpace.cpp`**

```cpp
// lib/AutoTuner/TilingSpace.cpp
#include "AutoTuner/TilingSpace.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <regex>
#include <sstream>

namespace mlir::autotuner {

// Minimal expression evaluator for constraint strings.
// Supports: integer literals, named variables, +, -, *, /, (, )
// No precedence parsing beyond the use of regex substitution.
bool TilingSpace::EvalConstraint(
    const std::string& expr,
    const std::vector<std::pair<std::string, int64_t>>& bindings) {

  // Split on <= / >= / ==
  auto splitOp = [](const std::string& s, const std::string& op)
      -> std::optional<std::pair<std::string, std::string>> {
    auto pos = s.find(op);
    if (pos == std::string::npos) return std::nullopt;
    return {{s.substr(0, pos), s.substr(pos + op.size())}};
  };

  // Substitute all variables by their integer values, then evaluate.
  auto substitute = [&](std::string s) -> std::string {
    // Sort by descending name length to avoid partial matches
    auto sorted = bindings;
    std::sort(sorted.begin(), sorted.end(),
              [](auto& a, auto& b) { return a.first.size() > b.first.size(); });
    for (auto& [name, val] : sorted) {
      std::string re = "\\b" + name + "\\b";
      s = std::regex_replace(s, std::regex(re), std::to_string(val));
    }
    return s;
  };

  // Evaluate a simple integer arithmetic expression (no variables remaining).
  // Self-contained: handles parentheses, +, -, *, / with correct precedence.
  std::function<int64_t(const std::string&)> eval = [&](const std::string& s) -> int64_t {
    std::string t;
    for (char c : s) if (c != ' ' && c != '\t') t += c;
    if (t.empty()) return 0;
    // Handle innermost parentheses first
    auto lp = t.rfind('(');
    if (lp != std::string::npos) {
      auto rp = t.find(')', lp);
      std::string inner = std::to_string(eval(t.substr(lp + 1, rp - lp - 1)));
      return eval(t.substr(0, lp) + inner + t.substr(rp + 1));
    }
    // + and - have lowest precedence (scan right-to-left)
    for (int i = (int)t.size()-1; i>=1; --i) {
      if (t[i]=='+') return eval(t.substr(0,i)) + eval(t.substr(i+1));
      if (t[i]=='-') return eval(t.substr(0,i)) - eval(t.substr(i+1));
    }
    // * and / have higher precedence
    for (int i = (int)t.size()-1; i>=1; --i) {
      if (t[i]=='*') return eval(t.substr(0,i)) * eval(t.substr(i+1));
      if (t[i]=='/') {
        int64_t d = eval(t.substr(i+1));
        return d != 0 ? eval(t.substr(0,i)) / d : 0;
      }
    }
    return std::stoll(t);
  };

  std::string subst = substitute(expr);

  if (auto parts = splitOp(subst, "<="))
    return eval(parts->first) <= eval(parts->second);
  if (auto parts = splitOp(subst, ">="))
    return eval(parts->first) >= eval(parts->second);
  if (auto parts = splitOp(subst, "=="))
    return eval(parts->first) == eval(parts->second);
  return true; // unknown constraint format → pass through
}

// Check if a constraint string is linear (no variable*variable products).
static bool isLinearConstraint(const std::string& expr,
                                const std::vector<std::string>& param_names) {
  // A product is nonlinear if it contains two variable names separated by *
  // Simple heuristic: count occurrences of param names in a product context
  std::string lower = expr;
  int var_count_in_product = 0;
  auto pos = expr.find('*');
  while (pos != std::string::npos) {
    // Check if both sides of * contain a variable name
    auto before = expr.substr(0, pos);
    auto after  = expr.substr(pos + 1);
    bool left_has_var  = false, right_has_var = false;
    for (auto& n : param_names) {
      if (before.find(n) != std::string::npos) left_has_var = true;
      if (after.find(n) != std::string::npos)  right_has_var = true;
    }
    if (left_has_var && right_has_var) return false;
    pos = expr.find('*', pos + 1);
  }
  return true;
}

static llvm::Expected<TilingSpace> loadFromPath(
    const std::string& path, const HardwareProfile& hw) {
  auto buf = llvm::MemoryBuffer::getFile(path);
  if (!buf)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot read %s", path.c_str());
  auto json = llvm::json::parse((*buf)->getBuffer());
  if (!json)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "JSON parse error in %s", path.c_str());
  const llvm::json::Object* obj = json->getAsObject();
  if (!obj)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Expected JSON object");

  TilingSpace ts;
  // Parse params
  if (auto* params_arr = obj->getArray("params")) {
    for (const auto& p : *params_arr) {
      const llvm::json::Object* po = p.getAsObject();
      if (!po) continue;
      TilingParam tp;
      tp.name      = po->getString("name").value_or("").str();
      tp.min       = po->getInteger("min").value_or(1);
      tp.max       = po->getInteger("max").value_or(128);
      tp.step      = po->getInteger("step").value_or(1);
      tp.alignment = po->getInteger("alignment").value_or(1);
      ts.params.push_back(tp);
    }
  }

  // Collect param names for linearity check
  std::vector<std::string> param_names;
  for (auto& p : ts.params) param_names.push_back(p.name);

  // Substitute hw constants into constraints
  std::vector<std::pair<std::string, int64_t>> hw_bindings = {
    {"ub_size", hw.ub_size},
    {"aiv_num", static_cast<int64_t>(hw.aiv_num)},
  };

  auto substituteHw = [&](std::string s) -> std::string {
    for (auto& [name, val] : hw_bindings) {
      std::string re = "\\b" + name + "\\b";
      s = std::regex_replace(s, std::regex(re), std::to_string(val));
    }
    return s;
  };

  // Initialize IntegerRelation with param_count dimensions
  ts.linear_constraints = mlir::presburger::IntegerRelation(
      mlir::presburger::PresburgerSpace::getSetSpace(
          static_cast<unsigned>(ts.params.size())));

  if (auto* cstr_arr = obj->getArray("constraints")) {
    for (const auto& c : *cstr_arr) {
      std::string expr = c.getAsString().value_or("").str();
      std::string subst = substituteHw(expr);

      if (!isLinearConstraint(subst, param_names)) {
        ts.nonlinear_exprs.push_back(subst);
      } else {
        // Linear constraint: parse into IntegerRelation for fast pruning.
        // Format after hw substitution: "<var> <= <var_or_const>" or similar.
        // Simple parser: handle "A <= B", "A >= B", "A == B" where A/B are
        // linear combinations of param names (index by position in ts.params).
        // For constraints involving shape vars (e.g., "TB_N <= N"), shape vars
        // are added as free dimensions to linear_constraints but won't be
        // checked during enumeration (their values come from shape_dims at runtime).
        // For correctness in early-exit pruning, we only add purely-param constraints.
        // Shape-variable constraints fall through to nonlinear_exprs (pointwise).
        bool has_shape_var = false;
        // Shape vars are names that are NOT in param_names and NOT all-digits
        // Heuristic: if after substituting param names with "0" the expression
        // still has alphabetic tokens, it references shape vars.
        std::string test_expr = subst;
        for (auto& n : param_names) {
          std::string re = "\\b" + n + "\\b";
          test_expr = std::regex_replace(test_expr, std::regex(re), "0");
        }
        std::regex alpha_re("[a-zA-Z]");
        if (std::regex_search(test_expr, alpha_re)) {
          has_shape_var = true;
        }
        if (has_shape_var) {
          // Contains shape variables (M, N, etc.) — route to pointwise eval
          ts.nonlinear_exprs.push_back(subst);
        } else {
          // Pure-param linear constraint — add to IntegerRelation.
          // Parse: find operator, split lhs/rhs into linear combination.
          // Variables are indexed by position in ts.params.
          auto tryAdd = [&](const std::string& lhs_s, const std::string& rhs_s,
                             int sign) -> bool {
            // Build coefficient vector: [params..., constant] (sign: lhs - rhs <= 0 or >= 0)
            // sign=1 means lhs <= rhs  →  lhs - rhs <= 0
            // sign=-1 means lhs >= rhs → rhs - lhs <= 0
            const unsigned nv = static_cast<unsigned>(ts.params.size());
            llvm::SmallVector<int64_t> coeffs(nv + 1, 0);
            // For each side, tokenize: look for "+"/"-" separated terms
            auto parseSide = [&](const std::string& s, int multiplier) {
              // Only handles integer literals and single param names for now
              for (unsigned i = 0; i < ts.params.size(); ++i) {
                if (s.find(ts.params[i].name) != std::string::npos) {
                  coeffs[i] += multiplier;
                  return;
                }
              }
              // Constant
              try {
                int64_t v = std::stoll(s);
                coeffs[nv] += multiplier * v;
              } catch (...) {}
            };
            parseSide(lhs_s, sign);
            parseSide(rhs_s, -sign);
            ts.linear_constraints.addInequality(coeffs);
            return true;
          };
          auto pos_le = subst.find("<=");
          auto pos_ge = subst.find(">=");
          auto pos_eq = subst.find("==");
          if (pos_le != std::string::npos) {
            tryAdd(subst.substr(0, pos_le), subst.substr(pos_le + 2), 1);
          } else if (pos_ge != std::string::npos) {
            tryAdd(subst.substr(0, pos_ge), subst.substr(pos_ge + 2), -1);
          } else if (pos_eq != std::string::npos) {
            tryAdd(subst.substr(0, pos_eq), subst.substr(pos_eq + 2), 1);
            tryAdd(subst.substr(0, pos_eq), subst.substr(pos_eq + 2), -1);
          } else {
            ts.nonlinear_exprs.push_back(subst); // fallback
          }
        }
      }
    }
  }

  return ts;
}

llvm::Expected<TilingSpace> TilingSpace::LoadFromJson(
    const std::string& func_name,
    const std::string& search_dir,
    const HardwareProfile& hw,
    const std::string& explicit_path) {

  // Priority 1: explicit path
  if (!explicit_path.empty()) return loadFromPath(explicit_path, hw);

  // Priority 2: {func_name}_tiling_space.json
  if (!func_name.empty()) {
    llvm::SmallString<256> p(search_dir);
    llvm::sys::path::append(p, func_name + "_tiling_space.json");
    if (llvm::sys::fs::exists(p)) return loadFromPath(p.str().str(), hw);
  }

  // Priority 3: tiling_space.json
  llvm::SmallString<256> p(search_dir);
  llvm::sys::path::append(p, "tiling_space.json");
  if (llvm::sys::fs::exists(p)) return loadFromPath(p.str().str(), hw);

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
      "No tiling_space.json found for func '%s' in %s",
      func_name.c_str(), search_dir.c_str());
}

} // namespace mlir::autotuner
```

- [ ] **Step 3: Commit**

```bash
git add include/AutoTuner/TilingSpace.h lib/AutoTuner/TilingSpace.cpp
git commit -m "feat(autotuner): add TilingSpace with dual-path constraint handling"
```

---

## Task 3: MlirAnalyzer

**Files:**
- Create: `include/AutoTuner/MlirAnalyzer.h`
- Modify: `lib/AutoTuner/MlirAnalyzer.cpp`

Background: In step7 IR, the TilingData parameter looks like:
```mlir
%tiling_data: memref<?x!emitasc.py_struct<"TilingData",
    [i64, i64, i64, i64],
    ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, 22 : i32>
```
Field naming convention:
- `"TB_*"` → TilingParam (cut tiling parameter, value from Solver)
- `"dim_argX_Y"` → ShapeDim (shape of arg X, dimension Y)

`data_copy_l2` ops carry the copy count as their third operand. The count is typically `%inner_size = arith.minsi(%remaining, %tb_n)` or `arith.minsi(%tb_m, %remaining)`. We take the RHS operand of minsi as a heuristic (the tile-size operand traces back via `arith.index_cast` → `emitasc.member` to the TilingData field; if RHS doesn't trace, the count expr falls back to AffineConstantExpr(1) safely).

`ascendc.get_block_idx` is followed by `arith.muli %block_idx, %tb_m`; the `%tb_m` operand identifies the parallelism-axis tiling parameter.

- [ ] **Step 1: Write `include/AutoTuner/MlirAnalyzer.h`**

```cpp
// include/AutoTuner/MlirAnalyzer.h
#pragma once
#include "llvm/Support/Error.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include <string>
#include <vector>
#include <map>

namespace mlir {
class ModuleOp;
class func::FuncOp;
}

namespace mlir::autotuner {

struct TilingField {
  std::string name;
  enum class Role { TilingParam, ShapeDim } role;
  int32_t field_index = 0;   // position in TilingData struct (0-based)
  int32_t arg_index   = -1;  // for ShapeDim: which func arg
  int32_t dim_index   = -1;  // for ShapeDim: which memref dimension
};

struct KernelAnalysis {
  std::string kernel_name;
  mlir::MLIRContext* ctx = nullptr;

  std::vector<TilingField> tiling_fields;  // in struct field order

  // shape_dims[i] = AffineDimExpr(i) corresponding to a dynamic memref dimension.
  // Parallel to a flat list of all dynamic dims across all func args.
  std::vector<mlir::AffineExpr> shape_dims;

  // Maps shape variable name (e.g., "M", "N" derived from dim_arg field names) to
  // its AffineDimExpr index in shape_dims.
  std::map<std::string, unsigned> shape_dim_index;

  struct PipeAccess {
    mlir::AffineExpr gm_to_ub;   // MTE2 byte count per core per iteration (upper bound)
    mlir::AffineExpr ub_to_gm;   // MTE3 byte count per core per iteration
    mlir::AffineExpr vec_flops;  // VEC flops (AffineConstantExpr(0) for now)
  } pipe_access;

  // block_dim = ceildiv(parallel_dim, TB_param)
  // parallel_dim is AffineDimExpr, TB_param is AffineSymbolExpr
  mlir::AffineExpr block_dim_expr;

  // Names of tiling params in order (extracted from tiling_fields Role==TilingParam)
  std::vector<std::string> tiling_param_names;

  std::vector<std::string> op_types;
};

class MlirAnalyzer {
public:
  // Analyze a single func.func (must have ascendc.aicore attribute).
  // Note: spec §5.2 declares Analyze(ModuleOp) but per-func is intentional here —
  // AutoTunerPass iterates funcs and calls Analyze per-func, which is cleaner.
  static llvm::Expected<KernelAnalysis> Analyze(mlir::func::FuncOp func);
};

} // namespace mlir::autotuner
```

- [ ] **Step 2: Write `lib/AutoTuner/MlirAnalyzer.cpp`**

```cpp
// lib/AutoTuner/MlirAnalyzer.cpp
#include "AutoTuner/MlirAnalyzer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Error.h"
#include <regex>

// Forward declare emitasc types used — replace with actual includes from externals/pyasc
// if PyStructType is available as a C++ type. Otherwise use attribute string inspection.

namespace mlir::autotuner {

// Parse field names from emitasc.py_struct type attribute string.
// Format: !emitasc.py_struct<"TilingData", [i64, i64, ...], ["TB_M", "TB_N", ...]>
static std::vector<std::string> parseFieldNames(const std::string& type_str) {
  std::vector<std::string> names;
  // Find the second list [...] which contains field names in quotes
  auto pos = type_str.rfind('[');
  if (pos == std::string::npos) return names;
  std::string fields_part = type_str.substr(pos);
  std::regex name_re(R"("([^"]+)")");
  auto it  = std::sregex_iterator(fields_part.begin(), fields_part.end(), name_re);
  auto end = std::sregex_iterator();
  for (; it != end; ++it) names.push_back((*it)[1].str());
  return names;
}

// Derive shape variable name from field name "dim_argX_Y" → e.g., "arg0_dim0"
// We use a simple label: argX_Y becomes symbolic dim name.
static std::string shapeVarName(int arg_idx, int dim_idx) {
  return "arg" + std::to_string(arg_idx) + "_dim" + std::to_string(dim_idx);
}

llvm::Expected<KernelAnalysis> MlirAnalyzer::Analyze(mlir::func::FuncOp func) {
  KernelAnalysis ka;
  ka.kernel_name = func.getName().str();
  ka.ctx = func.getContext();

  mlir::MLIRContext* ctx = ka.ctx;
  unsigned next_dim = 0;

  // ---- 1. Find TilingData parameter and extract field names ----
  for (auto arg : func.getArguments()) {
    std::string type_str;
    llvm::raw_string_ostream os(type_str);
    arg.getType().print(os);
    os.flush();

    if (type_str.find("py_struct") == std::string::npos ||
        type_str.find("TilingData") == std::string::npos)
      continue;

    auto field_names = parseFieldNames(type_str);
    for (int fi = 0; fi < (int)field_names.size(); ++fi) {
      const std::string& name = field_names[fi];
      TilingField tf;
      tf.name = name;
      tf.field_index = fi;

      if (name.rfind("TB_", 0) == 0 || name.rfind("Tb_", 0) == 0) {
        tf.role = TilingField::Role::TilingParam;
      } else if (name.rfind("dim_arg", 0) == 0) {
        tf.role = TilingField::Role::ShapeDim;
        // dim_argX_Y → arg_index=X, dim_index=Y
        std::regex re(R"(dim_arg(\d+)_(\d+))");
        std::smatch m;
        if (std::regex_match(name, m, re)) {
          tf.arg_index = std::stoi(m[1].str());
          tf.dim_index = std::stoi(m[2].str());
        }
      } else {
        tf.role = TilingField::Role::TilingParam; // default
      }
      ka.tiling_fields.push_back(tf);
    }
    break; // only one TilingData arg
  }

  // ---- 2. Build shape_dims from func args with dynamic memref dims ----
  // Map (arg_index, dim_index) → AffineDimExpr
  std::map<std::pair<int,int>, mlir::AffineExpr> dim_map;
  for (auto arg : func.getArguments()) {
    int arg_idx = arg.getArgNumber();
    auto memref_type = arg.getType().dyn_cast<mlir::MemRefType>();
    if (!memref_type) continue;
    for (int di = 0; di < (int)memref_type.getRank(); ++di) {
      if (memref_type.isDynamicDim(di)) {
        auto expr = mlir::getAffineDimExpr(next_dim, ctx);
        dim_map[{arg_idx, di}] = expr;
        std::string var_name = shapeVarName(arg_idx, di);
        ka.shape_dim_index[var_name] = next_dim;
        ka.shape_dims.push_back(expr);
        ++next_dim;
      }
    }
  }

  // Collect tiling param names
  for (auto& tf : ka.tiling_fields)
    if (tf.role == TilingField::Role::TilingParam)
      ka.tiling_param_names.push_back(tf.name);

  // Build AffineSymbolExpr for each tiling param (index by position in tiling_param_names)
  auto getParamSymbol = [&](const std::string& name) -> mlir::AffineExpr {
    for (unsigned i = 0; i < ka.tiling_param_names.size(); ++i)
      if (ka.tiling_param_names[i] == name)
        return mlir::getAffineSymbolExpr(i, ctx);
    return mlir::getAffineConstantExpr(0, ctx);
  };

  // ---- 3. Build pipe_access expressions from data_copy_l2 ops ----
  mlir::AffineExpr gm_ub = mlir::getAffineConstantExpr(0, ctx);
  mlir::AffineExpr ub_gm = mlir::getAffineConstantExpr(0, ctx);

  func.walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "ascendc.data_copy_l2") {
      // Operand 0: dst, 1: src, 2: count
      if (op->getNumOperands() < 3) return;
      mlir::Value count_val = op->getOperand(2);

      // Approximate count: if count comes from arith.minsi, take the tile-size operand.
      // Trace through index_cast ops.
      auto trace = [](mlir::Value v) -> mlir::Value {
        while (v) {
          if (auto cast = v.getDefiningOp<mlir::arith::IndexCastOp>())
            v = cast.getIn();
          else if (auto cast2 = v.getDefiningOp<mlir::arith::IndexCastUIOp>())
            v = cast2.getIn();
          else break;
        }
        if (auto minsi = v.getDefiningOp<mlir::arith::MinSIOp>()) {
          // In step7 IR, minsi patterns are:
          //   arith.minsi %remaining, %tb_n  (TB is RHS)
          //   arith.minsi %tb_m, %remaining  (TB is LHS)
          // Try RHS first (most common), then LHS. Heuristic: TB values come from
          // emitasc.member ops; non-TB values (remaining counts) come from arith ops.
          // Return both and let the emitasc.member trace below resolve which is TB.
          // For simplicity, return RHS first; if it doesn't trace to emitasc.member,
          // the count_expr stays AffineConstantExpr(1), which is still safe.
          return minsi.getRhs();
        }
        return v;
      };
      mlir::Value tile_val = trace(count_val);

      // Try to map tile_val to a TilingData field name via emitasc.member
      mlir::AffineExpr count_expr = mlir::getAffineConstantExpr(1, ctx);
      // Walk def chain for emitasc.member
      mlir::Value v = tile_val;
      while (v && v.getDefiningOp()) {
        auto* def = v.getDefiningOp();
        if (def->getName().getStringRef() == "emitasc.member") {
          // EmitAsc_MemberOp declares the field name as StrAttr:$field, so
          // the attribute key is "field" — use getAttr directly to avoid
          // accidentally matching other StringAttrs on the op.
          if (auto sa = def->getAttrOfType<mlir::StringAttr>("field")) {
            std::string fname = sa.str();
            for (auto& pname : ka.tiling_param_names)
              if (fname == pname) {
                count_expr = getParamSymbol(pname);
                goto done_trace;
              }
          }
        }
        // Follow single result chains
        if (def->getNumOperands() == 1)
          v = def->getOperand(0);
        else if (def->getNumOperands() == 2)
          v = def->getOperand(1); // for index_cast etc.
        else break;
      }
      done_trace:;

      // Element size: 2 bytes for f16 (assume f16 for now)
      auto two = mlir::getAffineConstantExpr(2, ctx);
      mlir::AffineExpr bytes = count_expr * two;

      // Determine direction: check if dst is GlobalTensor (GM→UB) or src is GlobalTensor (UB→GM)
      // Heuristic: first operand's type name contains "global_tensor" → src=UB, dst=GM (UB→GM)
      //            second operand contains "global_tensor" → src=GM, dst=UB (GM→UB)
      std::string dst_type_str, src_type_str;
      {
        llvm::raw_string_ostream os1(dst_type_str);
        op->getOperand(0).getType().print(os1);
        llvm::raw_string_ostream os2(src_type_str);
        op->getOperand(1).getType().print(os2);
      }
      bool dst_is_global = dst_type_str.find("global_tensor") != std::string::npos;
      bool src_is_global = src_type_str.find("global_tensor") != std::string::npos;

      if (src_is_global) gm_ub = gm_ub + bytes;
      if (dst_is_global) ub_gm = ub_gm + bytes;
    }
  });

  ka.pipe_access.gm_to_ub = gm_ub;
  ka.pipe_access.ub_to_gm = ub_gm;
  ka.pipe_access.vec_flops = mlir::getAffineConstantExpr(0, ctx);

  // ---- 4. block_dim_expr ----
  // Find the TilingParam that's used as multiplier of get_block_idx
  // (i.e., %block_offset = arith.muli %block_idx, %tb_m)
  std::string parallel_param_name;
  func.walk([&](mlir::arith::MulIOp mul) {
    if (parallel_param_name.empty()) {
      for (auto operand : mul.getOperands()) {
        auto* def = operand.getDefiningOp();
        if (def && def->getName().getStringRef() == "ascendc.get_block_idx") {
          // Other operand is the TB param
          mlir::Value other = (mul.getLhs() == operand) ? mul.getRhs() : mul.getLhs();
          // Trace to emitasc.member
          mlir::Value v = other;
          while (v && v.getDefiningOp()) {
            auto* d = v.getDefiningOp();
            if (d->getName().getStringRef() == "emitasc.member") {
              // Use getAttr("field") directly — EmitAsc_MemberOp's StrAttr:$field
              if (auto sa = d->getAttrOfType<mlir::StringAttr>("field")) {
                parallel_param_name = sa.str();
                return;
              }
            }
            if (d->getNumOperands() == 1) v = d->getOperand(0);
            else break;
          }
          return;
        }
      }
    }
  });

  if (!parallel_param_name.empty()) {
    // Find corresponding shape dim from ShapeDim field with matching arg_index
    // (the TB_M param controls the M dimension → find "dim_argX_0" where X is the arg
    //  whose 0th dim is M)
    // Heuristic: find TilingParam field index, then look for ShapeDim field with
    // arg_index derived from parallel_param_name suffix or just use first ShapeDim.
    mlir::AffineExpr parallel_dim_expr = mlir::getAffineConstantExpr(1, ctx);
    for (auto& tf : ka.tiling_fields) {
      if (tf.role == TilingField::Role::ShapeDim && tf.dim_index == 0) {
        auto key = shapeVarName(tf.arg_index, tf.dim_index);
        if (ka.shape_dim_index.count(key))
          parallel_dim_expr = ka.shape_dims[ka.shape_dim_index.at(key)];
        break;
      }
    }
    auto tb_sym = getParamSymbol(parallel_param_name);
    // ceildiv(M, TB_M) = (M + TB_M - 1) / TB_M
    auto one = mlir::getAffineConstantExpr(1, ctx);
    ka.block_dim_expr = (parallel_dim_expr + tb_sym - one).floorDiv(tb_sym);
  } else {
    ka.block_dim_expr = mlir::getAffineConstantExpr(1, ctx);
  }

  // ---- 5. op_types ----
  func.walk([&](mlir::Operation* op) {
    auto name = op->getName().getStringRef().str();
    if (name.rfind("ascendc.", 0) == 0)
      ka.op_types.push_back(name);
  });

  return ka;
}

} // namespace mlir::autotuner
```

- [ ] **Step 3: Commit**

```bash
git add include/AutoTuner/MlirAnalyzer.h lib/AutoTuner/MlirAnalyzer.cpp
git commit -m "feat(autotuner): add MlirAnalyzer (step7 IR extraction)"
```

---

## Task 4: Solver (EnumerateSolver)

**Files:**
- Create: `include/AutoTuner/Solver.h`
- Modify: `lib/AutoTuner/Solver.cpp`

- [ ] **Step 1: Write `include/AutoTuner/Solver.h`**

```cpp
// include/AutoTuner/Solver.h
#pragma once
#include "AutoTuner/HardwareProfile.h"
#include "AutoTuner/MlirAnalyzer.h"
#include "AutoTuner/TilingSpace.h"
#include "llvm/Support/Error.h"
#include "mlir/IR/AffineExpr.h"
#include <map>
#include <string>

namespace mlir::autotuner {

struct TilingExprResult {
  // TilingParam name → its expression (AffineConstantExpr for EnumerateSolver)
  std::map<std::string, mlir::AffineExpr> param_exprs;

  // Performance estimate: static_cast<double>(numerator) / bandwidth_gbs
  // numerator: AffineExpr with shape symbols (AffineDimExpr), tiling params substituted
  struct PerfExpr {
    mlir::AffineExpr numerator;     // integer byte-count expression
    double           bandwidth_gbs; // HW bandwidth constant
  } perf_expr;

  // block_dim expression (tiling params substituted; shape dims remain as AffineDimExpr)
  mlir::AffineExpr block_dim_expr;
};

class SolverBase {
public:
  virtual ~SolverBase() = default;
  virtual llvm::Expected<TilingExprResult>
  Solve(const KernelAnalysis& analysis,
        const TilingSpace&    space,
        const HardwareProfile& hw) = 0;
};

// Enumerate all param combinations, prune by constraints, score by MTE2 bandwidth.
class EnumerateSolver : public SolverBase {
public:
  llvm::Expected<TilingExprResult>
  Solve(const KernelAnalysis& analysis,
        const TilingSpace&    space,
        const HardwareProfile& hw) override;
};

} // namespace mlir::autotuner
```

- [ ] **Step 2: Write `lib/AutoTuner/Solver.cpp`**

```cpp
// lib/AutoTuner/Solver.cpp
#include "AutoTuner/Solver.h"
#include "mlir/IR/AffineExpr.h"
#include <cassert>
#include <limits>
#include <vector>

namespace mlir::autotuner {

// Evaluate an AffineExpr with concrete values for dims and symbols.
static int64_t evalAffine(mlir::AffineExpr expr,
                           llvm::ArrayRef<int64_t> dims,
                           llvm::ArrayRef<int64_t> syms) {
  switch (expr.getKind()) {
    case mlir::AffineExprKind::Constant:
      return expr.cast<mlir::AffineConstantExpr>().getValue();
    case mlir::AffineExprKind::DimId:
      return dims[expr.cast<mlir::AffineDimExpr>().getPosition()];
    case mlir::AffineExprKind::SymbolId:
      return syms[expr.cast<mlir::AffineSymbolExpr>().getPosition()];
    case mlir::AffineExprKind::Add: {
      auto b = expr.cast<mlir::AffineBinaryOpExpr>();
      return evalAffine(b.getLHS(), dims, syms) + evalAffine(b.getRHS(), dims, syms);
    }
    case mlir::AffineExprKind::Mul: {
      auto b = expr.cast<mlir::AffineBinaryOpExpr>();
      return evalAffine(b.getLHS(), dims, syms) * evalAffine(b.getRHS(), dims, syms);
    }
    case mlir::AffineExprKind::FloorDiv: {
      auto b = expr.cast<mlir::AffineBinaryOpExpr>();
      int64_t d = evalAffine(b.getRHS(), dims, syms);
      if (d == 0) return 0;
      int64_t n = evalAffine(b.getLHS(), dims, syms);
      return n / d; // floor div for non-negative values
    }
    case mlir::AffineExprKind::CeilDiv: {
      auto b = expr.cast<mlir::AffineBinaryOpExpr>();
      int64_t d = evalAffine(b.getRHS(), dims, syms);
      if (d == 0) return 0;
      int64_t n = evalAffine(b.getLHS(), dims, syms);
      return (n + d - 1) / d;
    }
    case mlir::AffineExprKind::Mod: {
      auto b = expr.cast<mlir::AffineBinaryOpExpr>();
      int64_t d = evalAffine(b.getRHS(), dims, syms);
      if (d == 0) return 0;
      return evalAffine(b.getLHS(), dims, syms) % d;
    }
    default: return 0;
  }
}

// Substitute symbol expressions into an AffineExpr.
// param_vals[i] = concrete value for symbol i (in analysis.tiling_param_names order).
// Returns AffineExpr with only dims remaining (symbols replaced by constants).
static mlir::AffineExpr substituteSymbols(
    mlir::AffineExpr expr, const std::vector<int64_t>& sym_vals) {

  mlir::MLIRContext* ctx = expr.getContext();
  switch (expr.getKind()) {
    case mlir::AffineExprKind::Constant:
    case mlir::AffineExprKind::DimId:
      return expr;
    case mlir::AffineExprKind::SymbolId: {
      unsigned pos = expr.cast<mlir::AffineSymbolExpr>().getPosition();
      if (pos < sym_vals.size())
        return mlir::getAffineConstantExpr(sym_vals[pos], ctx);
      return expr;
    }
    default: {
      auto b = expr.cast<mlir::AffineBinaryOpExpr>();
      auto lhs = substituteSymbols(b.getLHS(), sym_vals);
      auto rhs = substituteSymbols(b.getRHS(), sym_vals);
      switch (expr.getKind()) {
        case mlir::AffineExprKind::Add:      return lhs + rhs;
        case mlir::AffineExprKind::Mul:      return lhs * rhs;
        case mlir::AffineExprKind::FloorDiv: return lhs.floorDiv(rhs);
        case mlir::AffineExprKind::CeilDiv:  return lhs.ceilDiv(rhs);
        case mlir::AffineExprKind::Mod:      return lhs % rhs;
        default: return expr;
      }
    }
  }
}

llvm::Expected<TilingExprResult> EnumerateSolver::Solve(
    const KernelAnalysis& analysis,
    const TilingSpace&    space,
    const HardwareProfile& hw) {

  if (space.params.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "TilingSpace has no params");

  mlir::MLIRContext* ctx = analysis.ctx;

  // Build candidate list via Cartesian product of param ranges.
  // Each candidate = vector<int64_t> of param values in space.params order.
  std::vector<std::vector<int64_t>> candidates = {{}};
  for (auto& p : space.params) {
    std::vector<std::vector<int64_t>> next;
    for (int64_t v = p.min; v <= p.max; v += p.step) {
      if (p.alignment > 0 && v % p.alignment != 0) continue;
      for (auto c : candidates) {
        c.push_back(v);
        next.push_back(c);
      }
    }
    candidates = std::move(next);
  }

  // Prune by nonlinear constraints.
  // Shape dims are unknown at compile time → use placeholder 1024 for pruning
  // (conservative: constraints like TB_N <= N pass for any reasonable N).
  // Actual shape-dependent constraints (like TB_M <= M) are checked at runtime.
  std::vector<std::pair<std::string, int64_t>> prune_bindings;
  prune_bindings.push_back({"ub_size", hw.ub_size});
  prune_bindings.push_back({"aiv_num", static_cast<int64_t>(hw.aiv_num)});
  // Shape dims: use large placeholder so shape-bound constraints don't falsely prune
  for (auto& [name, idx] : analysis.shape_dim_index)
    prune_bindings.push_back({name, 65536}); // placeholder

  std::vector<std::vector<int64_t>> valid;
  for (auto& cand : candidates) {
    // Add param values to bindings
    auto bindings = prune_bindings;
    for (size_t i = 0; i < space.params.size(); ++i)
      bindings.push_back({space.params[i].name, cand[i]});

    bool ok = true;
    for (auto& expr : space.nonlinear_exprs) {
      if (!TilingSpace::EvalConstraint(expr, bindings)) { ok = false; break; }
    }
    if (ok) valid.push_back(cand);
  }

  if (valid.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No valid tiling candidates after constraint pruning");

  // Score candidates by estimated total MTE2 bytes:
  //   total = gm_to_ub_per_call(TB) * trip_count(TB, shape)
  // For fixed shapes (dummy=1), trip_count = ceildiv(1, TB_M) * ceildiv(1, TB_N) ≈ 1/(TB_M*TB_N)
  // To compare candidates fairly, use total_bytes = per_call_bytes * trip_count.
  // With shape dims = S (const S for ranking purposes), this factors out, so:
  //   score = gm_to_ub_per_call / (TB_M * TB_N * ...)
  // Larger tile → fewer trips → lower score → better.
  // Use shape dims = large constant (e.g., 1024) so shape-dependent terms scale.
  std::vector<int64_t> dummy_dims(analysis.shape_dims.size(), 1024);
  int best_idx = 0;
  // Use ratio: per_call_bytes * num_blocks; num_blocks = ceildiv(shape, TB_parallel)
  // Simplified: score = per_call_bytes * (1 / TB_parallel) → minimize per_call / TB_par
  // For enumeration purposes, just pick candidate with highest per-call throughput:
  // score = per_call_bytes (lower is NOT better — larger TB → more bytes per call but fewer calls)
  // Correct metric: total = per_call * ceil(S/TB) ≈ per_call * (S/TB)
  // For fixed S, this is proportional to per_call / TB.
  // Since gm_to_ub typically contains TB as a linear factor (e.g., TB_N * S * 2 bytes),
  // total ∝ TB_N * S * 2 * (S / TB_N) = S^2 * 2 — independent of TB_N for this pattern.
  // In practice, per_call = TB_M * TB_N * 2, trips = ceil(M/TB_M) * ceil(N/TB_N),
  // total ≈ M * N * 2 — same for all candidates. So any valid candidate is equivalent.
  // Pick largest TB values (fewest memory transactions, lowest overhead).
  int64_t best_score = std::numeric_limits<int64_t>::min();

  for (int ci = 0; ci < (int)valid.size(); ++ci) {
    const auto& cand = valid[ci];
    // Build sym_vals in tiling_param_names order
    std::vector<int64_t> sym_vals(analysis.tiling_param_names.size(), 1);
    for (size_t pi = 0; pi < space.params.size(); ++pi) {
      for (size_t si = 0; si < analysis.tiling_param_names.size(); ++si) {
        if (analysis.tiling_param_names[si] == space.params[pi].name) {
          sym_vals[si] = cand[pi];
          break;
        }
      }
    }
    // Score = per-call bytes (with shape=1024). Larger tile → higher per-call bytes → better.
    // This picks the candidate that moves the most data per call (fewest calls total).
    int64_t score = evalAffine(analysis.pipe_access.gm_to_ub, dummy_dims, sym_vals);
    if (score > best_score) { best_score = score; best_idx = ci; }
  }

  const auto& best = valid[best_idx];

  // Build TilingExprResult
  TilingExprResult result;
  for (size_t i = 0; i < space.params.size(); ++i)
    result.param_exprs[space.params[i].name] =
        mlir::getAffineConstantExpr(best[i], ctx);

  // Build sym_vals for symbol substitution
  std::vector<int64_t> sym_vals(analysis.tiling_param_names.size(), 1);
  for (size_t pi = 0; pi < space.params.size(); ++pi)
    for (size_t si = 0; si < analysis.tiling_param_names.size(); ++si)
      if (analysis.tiling_param_names[si] == space.params[pi].name) {
        sym_vals[si] = best[pi];
        break;
      }

  // perf_expr numerator: substitute tiling params, keep shape dims
  result.perf_expr.numerator   = substituteSymbols(analysis.pipe_access.gm_to_ub, sym_vals);
  result.perf_expr.bandwidth_gbs = hw.mte2_bandwidth;

  // block_dim_expr: substitute tiling params, keep shape dims
  result.block_dim_expr = substituteSymbols(analysis.block_dim_expr, sym_vals);

  return result;
}

} // namespace mlir::autotuner
```

- [ ] **Step 3: Commit**

```bash
git add include/AutoTuner/Solver.h lib/AutoTuner/Solver.cpp
git commit -m "feat(autotuner): add EnumerateSolver (enumerate+prune+score)"
```

---

## Task 5: TilingFuncEmitter

**Files:**
- Create: `include/AutoTuner/TilingFuncEmitter.h`
- Modify: `lib/AutoTuner/TilingFuncEmitter.cpp`

- [ ] **Step 1: Write `include/AutoTuner/TilingFuncEmitter.h`**

```cpp
// include/AutoTuner/TilingFuncEmitter.h
#pragma once
#include "AutoTuner/MlirAnalyzer.h"
#include "AutoTuner/Solver.h"
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace mlir::autotuner {

class TilingFuncEmitter {
public:
  // Emit tiling_func_{graph_id}.cpp into output_dir.
  llvm::Error EmitSingleGraphTilingFunc(
      const TilingExprResult& result,
      const KernelAnalysis&   analysis,
      int                     graph_id,
      const std::string&      output_dir);

  // Emit get_tiling.cpp into output_dir.
  // Single-graph: direct forward without score comparison.
  // Multi-graph: compare ScoreTemplate scores, dispatch to best.
  llvm::Error EmitGetTilingEntry(
      const std::vector<TilingExprResult>& results,
      const std::vector<KernelAnalysis>&   analyses,
      const std::string&                   output_dir);

  // Emit empty online-tuning stub (reserved interface).
  llvm::Error EmitOnlineTuningStub(
      const KernelAnalysis& analysis,
      const std::string&    output_dir);
};

} // namespace mlir::autotuner
```

- [ ] **Step 2: Write `lib/AutoTuner/TilingFuncEmitter.cpp`**

```cpp
// lib/AutoTuner/TilingFuncEmitter.cpp
#include "AutoTuner/TilingFuncEmitter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <sstream>

namespace mlir::autotuner {

// Convert AffineExpr to C++ expression string.
// dims[i] → shape variable names (e.g., "M", "N")
// symbols → not used in output (tiling params already substituted to constants)
static std::string affineToC(mlir::AffineExpr expr,
                              const std::vector<std::string>& dim_names) {
  switch (expr.getKind()) {
    case mlir::AffineExprKind::Constant:
      return std::to_string(expr.cast<mlir::AffineConstantExpr>().getValue());
    case mlir::AffineExprKind::DimId: {
      unsigned pos = expr.cast<mlir::AffineDimExpr>().getPosition();
      if (pos < dim_names.size()) return dim_names[pos];
      return "dim" + std::to_string(pos);
    }
    case mlir::AffineExprKind::SymbolId: {
      unsigned pos = expr.cast<mlir::AffineSymbolExpr>().getPosition();
      return "sym" + std::to_string(pos);
    }
    default: {
      auto b = expr.cast<mlir::AffineBinaryOpExpr>();
      std::string lhs = affineToC(b.getLHS(), dim_names);
      std::string rhs = affineToC(b.getRHS(), dim_names);
      switch (expr.getKind()) {
        case mlir::AffineExprKind::Add:
          return "(" + lhs + " + " + rhs + ")";
        case mlir::AffineExprKind::Mul:
          return "(" + lhs + " * " + rhs + ")";
        case mlir::AffineExprKind::FloorDiv:
          return "(" + lhs + " / " + rhs + ")";
        case mlir::AffineExprKind::CeilDiv:
          return "((" + lhs + " + " + rhs + " - 1) / " + rhs + ")";
        case mlir::AffineExprKind::Mod:
          return "(" + lhs + " % " + rhs + ")";
        default: return "0";
      }
    }
  }
}

// Build ordered shape param names for function signature.
// e.g., analysis.shape_dims → ["M", "N"] derived from tiling_fields ShapeDim entries.
static std::vector<std::string> buildShapeParamNames(const KernelAnalysis& analysis) {
  // Use tiling_fields to find ShapeDim entries in field order
  std::map<unsigned, std::string> dim_idx_to_name;
  for (auto& tf : analysis.tiling_fields) {
    if (tf.role == TilingField::Role::ShapeDim) {
      std::string var = "arg" + std::to_string(tf.arg_index) +
                        "_dim" + std::to_string(tf.dim_index);
      if (analysis.shape_dim_index.count(var)) {
        unsigned idx = analysis.shape_dim_index.at(var);
        // Use field name as hint: dim_arg0_0 → M style (just use argX_Y for clarity)
        dim_idx_to_name[idx] = var;
      }
    }
  }
  std::vector<std::string> names(analysis.shape_dims.size());
  for (auto& [idx, name] : dim_idx_to_name) {
    if (idx < names.size()) names[idx] = name;
  }
  // Fill gaps
  for (size_t i = 0; i < names.size(); ++i)
    if (names[i].empty()) names[i] = "dim" + std::to_string(i);
  return names;
}

llvm::Error TilingFuncEmitter::EmitSingleGraphTilingFunc(
    const TilingExprResult& result,
    const KernelAnalysis& analysis,
    int graph_id,
    const std::string& output_dir) {

  std::string filename = output_dir + "/tiling_func_" + std::to_string(graph_id) + ".cpp";
  std::ofstream f(filename);
  if (!f)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot write %s", filename.c_str());

  auto shape_names = buildShapeParamNames(analysis);

  f << "// Auto-generated by AutoTunerPass -- do not edit\n";
  f << "#include \"tiling_func_" << graph_id << ".h\"\n\n";

  // Build function signature shape params
  std::string sig_params;
  for (auto& n : shape_names) sig_params += "int64_t " + n + ", ";

  std::string func_name = "GetTiling_" + analysis.kernel_name + "_" + std::to_string(graph_id);

  f << "void " << func_name << "(\n";
  for (auto& n : shape_names) f << "    int64_t " << n << ",\n";
  f << "    int64_t* block_dim_out,\n";
  f << "    TilingData* tiling) {\n";

  // Emit tiling param constants
  for (auto& [name, expr] : result.param_exprs) {
    std::string val = affineToC(expr, shape_names);
    f << "  const int64_t " << name << " = " << val << ";\n";
  }

  // Emit block_dim
  f << "  *block_dim_out = " << affineToC(result.block_dim_expr, shape_names) << ";\n";

  // Emit TilingData field assignments in struct field order
  for (auto& tf : analysis.tiling_fields) {
    if (tf.role == TilingField::Role::TilingParam) {
      f << "  tiling->" << tf.name << " = " << tf.name << ";\n";
    } else { // ShapeDim
      // Value comes from the corresponding shape param
      std::string var = "arg" + std::to_string(tf.arg_index) +
                        "_dim" + std::to_string(tf.dim_index);
      if (analysis.shape_dim_index.count(var)) {
        unsigned idx = analysis.shape_dim_index.at(var);
        f << "  tiling->" << tf.name << " = " << shape_names[idx] << ";\n";
      }
    }
  }

  f << "}\n\n";

  // Emit ScoreTemplate (marked [[maybe_unused]] for single-graph build)
  std::string score_name = "ScoreTemplate_" + std::to_string(graph_id);
  f << "[[maybe_unused]]\n";
  f << "double " << score_name << "(";
  bool first = true;
  for (auto& n : shape_names) {
    if (!first) f << ", ";
    f << "int64_t " << n;
    first = false;
  }
  f << ") {\n";
  f << "  return static_cast<double>("
    << affineToC(result.perf_expr.numerator, shape_names)
    << ") / " << result.perf_expr.bandwidth_gbs << "e9;\n";
  f << "}\n";

  return llvm::Error::success();
}

llvm::Error TilingFuncEmitter::EmitGetTilingEntry(
    const std::vector<TilingExprResult>& results,
    const std::vector<KernelAnalysis>&   analyses,
    const std::string&                   output_dir) {

  if (results.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "No results to emit");

  std::string filename = output_dir + "/get_tiling.cpp";
  std::ofstream f(filename);
  if (!f)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot write %s", filename.c_str());

  const KernelAnalysis& first = analyses[0];
  auto shape_names = buildShapeParamNames(first);

  f << "// Auto-generated by AutoTunerPass -- do not edit\n";
  for (size_t i = 0; i < results.size(); ++i)
    f << "#include \"tiling_func_" << i << ".h\"\n";
  f << "\n";

  // Forward declare ScoreTemplate functions
  for (size_t i = 0; i < results.size(); ++i) {
    f << "double ScoreTemplate_" << i << "(";
    bool fst = true;
    for (auto& n : shape_names) { if (!fst) f << ", "; f << "int64_t " << n; fst = false; }
    f << ");\n";
  }
  f << "\n";

  // Entry function
  std::string entry_name = "GetTiling_" + first.kernel_name;
  f << "void " << entry_name << "(\n";
  for (auto& n : shape_names) f << "    int64_t " << n << ",\n";
  f << "    int* selected_kernel_id,\n";
  f << "    int64_t* block_dim,\n";
  f << "    TilingData* tiling) {\n";

  if (results.size() == 1) {
    // Single-graph: direct forward
    f << "  *selected_kernel_id = 0;\n";
    f << "  GetTiling_" << first.kernel_name << "_0(";
    for (auto& n : shape_names) f << n << ", ";
    f << "block_dim, tiling);\n";
  } else {
    // Multi-graph: score comparison
    f << "  double scores[] = {";
    for (size_t i = 0; i < results.size(); ++i) {
      if (i) f << ", ";
      f << "ScoreTemplate_" << i << "(";
      bool fst = true;
      for (auto& n : shape_names) { if (!fst) f << ", "; f << n; fst = false; }
      f << ")";
    }
    f << "};\n";
    f << "  int best = 0;\n";
    f << "  for (int i = 1; i < " << results.size() << "; ++i)\n";
    f << "    if (scores[i] < scores[best]) best = i;\n";
    f << "  *selected_kernel_id = best;\n";
    f << "  switch (best) {\n";
    for (size_t i = 0; i < results.size(); ++i) {
      f << "    case " << i << ": GetTiling_" << analyses[i].kernel_name << "_" << i << "(";
      for (auto& n : shape_names) f << n << ", ";
      f << "block_dim, tiling); break;\n";
    }
    f << "  }\n";
  }

  f << "}\n";
  return llvm::Error::success();
}

llvm::Error TilingFuncEmitter::EmitOnlineTuningStub(
    const KernelAnalysis& analysis,
    const std::string& output_dir) {
  std::string filename = output_dir + "/online_tuning_stub.cpp";
  std::ofstream f(filename);
  if (!f)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot write %s", filename.c_str());
  f << "// Online tuning stub -- reserved for future implementation\n";
  f << "// Kernel: " << analysis.kernel_name << "\n";
  return llvm::Error::success();
}

} // namespace mlir::autotuner
```

- [ ] **Step 3: Commit**

```bash
git add include/AutoTuner/TilingFuncEmitter.h lib/AutoTuner/TilingFuncEmitter.cpp
git commit -m "feat(autotuner): add TilingFuncEmitter (AffineExpr → C++ codegen)"
```

---

## Task 6: AutoTunerPass — TableGen + registration

**Files:**
- Create: `include/AutoTuner/AutoTunerPass.h`
- Modify: `lib/AutoTuner/AutoTunerPass.cpp`
- Modify: `include/Conversion/Passes.td` (add pass def)
- Modify: `include/Conversion/Passes.h` (add include)
- Modify: `lib/CMakeLists.txt` (add subdirectory — `lib/AutoTuner/` is sibling to `lib/Conversion/`)
- Modify: `tools/afir-opt/CMakeLists.txt` (link new lib)

- [ ] **Step 1: Add pass to `include/Conversion/Passes.td`**

Open `include/Conversion/Passes.td` and append before the closing `}` or at the end of the pass list:

```tablegen
def AutoTunerPass : Pass<"autotuner", "mlir::ModuleOp"> {
  let summary = "Analyze step7 kernel IR and emit Host C++ tiling functions";
  let constructor = "mlir::afir::createAutoTunerPass()";
  let options = [
    Option<"hwName",    "autotuner-hw",         "std::string", /*default=*/"\"Ascend910B1\"",
           "SoC hardware profile name">,
    Option<"hwDir",     "autotuner-hw-dir",      "std::string", /*default=*/"\"\"",
           "Directory containing hardware JSON profiles">,
    Option<"outputDir", "autotuner-output-dir",  "std::string", /*default=*/"\"./autotuner_out\"",
           "Output directory for generated tiling C++ files">,
    Option<"spacePath", "autotuner-space",       "std::string", /*default=*/"\"\"",
           "Explicit tiling_space.json path (overrides auto-discovery)">,
  ];
}
```

- [ ] **Step 2: Write `include/AutoTuner/AutoTunerPass.h`**

```cpp
// include/AutoTuner/AutoTunerPass.h
#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::afir {
std::unique_ptr<Pass> createAutoTunerPass();
} // namespace mlir::afir
```

- [ ] **Step 3: Add include to `include/Conversion/Passes.h`**

Open `include/Conversion/Passes.h` and add with other includes:
```cpp
#include "AutoTuner/AutoTunerPass.h"
```

- [ ] **Step 4: Write `lib/AutoTuner/AutoTunerPass.cpp`**

```cpp
// lib/AutoTuner/AutoTunerPass.cpp
#define GEN_PASS_DECL_AUTOTUNERPASS
#define GEN_PASS_DEF_AUTOTUNERPASS
#include "Conversion/Passes.h.inc"

#include "AutoTuner/AutoTunerPass.h"
#include "AutoTuner/HardwareProfile.h"
#include "AutoTuner/MlirAnalyzer.h"
#include "AutoTuner/Solver.h"
#include "AutoTuner/TilingFuncEmitter.h"
#include "AutoTuner/TilingSpace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FileSystem.h"

namespace mlir::afir {

struct AutoTunerPassImpl
    : public impl::AutoTunerPassBase<AutoTunerPassImpl> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Determine search_dir: directory of the module's file (or cwd if unknown)
    std::string search_dir = ".";
    if (auto loc = module.getLoc().dyn_cast<mlir::FileLineColLoc>()) {
      llvm::SmallString<256> p(loc.getFilename().str());
      llvm::sys::path::remove_filename(p);
      search_dir = p.str().str();
      if (search_dir.empty()) search_dir = ".";
    }

    // Load hardware profile
    auto hw_or = autotuner::HardwareProfile::Load(hwName, hwDir);
    if (!hw_or) {
      module.emitError("AutoTuner: failed to load hardware profile: " +
                       llvm::toString(hw_or.takeError()));
      return signalPassFailure();
    }
    auto hw = std::move(*hw_or);

    // Create output dir
    if (auto ec = llvm::sys::fs::create_directories(outputDir)) {
      module.emitError("AutoTuner: cannot create output dir: " + ec.message());
      return signalPassFailure();
    }

    // Process each aicore func
    std::vector<autotuner::TilingExprResult> all_results;
    std::vector<autotuner::KernelAnalysis>   all_analyses;
    int graph_id = 0;

    for (auto func : module.getOps<mlir::func::FuncOp>()) {
      // Skip non-aicore funcs (e.g., transform.named_sequence uses func.func too)
      if (!func->hasAttr("ascendc.aicore")) continue;

      auto analysis_or = autotuner::MlirAnalyzer::Analyze(func);
      if (!analysis_or) {
        func.emitError("AutoTuner: analysis failed: " +
                       llvm::toString(analysis_or.takeError()));
        return signalPassFailure();
      }
      auto analysis = std::move(*analysis_or);

      auto space_or = autotuner::TilingSpace::LoadFromJson(
          analysis.kernel_name, search_dir, hw, spacePath);
      if (!space_or) {
        func.emitError("AutoTuner: failed to load tiling space: " +
                       llvm::toString(space_or.takeError()));
        return signalPassFailure();
      }

      autotuner::EnumerateSolver solver;
      auto result_or = solver.Solve(analysis, *space_or, hw);
      if (!result_or) {
        func.emitError("AutoTuner: solver failed: " +
                       llvm::toString(result_or.takeError()));
        return signalPassFailure();
      }

      autotuner::TilingFuncEmitter emitter;
      if (auto err = emitter.EmitSingleGraphTilingFunc(
              *result_or, analysis, graph_id, outputDir)) {
        func.emitError("AutoTuner: codegen failed: " +
                       llvm::toString(std::move(err)));
        return signalPassFailure();
      }

      all_results.push_back(std::move(*result_or));
      all_analyses.push_back(std::move(analysis));
      ++graph_id;
    }

    if (all_results.empty()) return; // no aicore funcs found

    autotuner::TilingFuncEmitter emitter;
    if (auto err = emitter.EmitGetTilingEntry(all_results, all_analyses, outputDir)) {
      module.emitError("AutoTuner: get_tiling codegen failed: " +
                       llvm::toString(std::move(err)));
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createAutoTunerPass() {
  return std::make_unique<AutoTunerPassImpl>();
}

} // namespace mlir::afir
```

- [ ] **Step 5: Add to `lib/CMakeLists.txt`**

`lib/AutoTuner/` is a sibling of `lib/Conversion/`, not a subdirectory of it.
Find and add to `lib/CMakeLists.txt` (alongside the existing `add_subdirectory(Conversion)` line):
```cmake
add_subdirectory(AutoTuner)
```

- [ ] **Step 6: Add to `tools/afir-opt/CMakeLists.txt`**

In the `target_link_libraries` list, add:
```cmake
AutoTunerConversion
```

- [ ] **Step 7: Commit**

```bash
git add include/AutoTuner/AutoTunerPass.h lib/AutoTuner/AutoTunerPass.cpp
git add include/Conversion/Passes.td include/Conversion/Passes.h
git add lib/CMakeLists.txt tools/afir-opt/CMakeLists.txt
git commit -m "feat(autotuner): register AutoTunerPass via TableGen"
```

---

## Task 7: Build and integration test

- [ ] **Step 1: Create `examples/broadcast-add-reduce/tiling_space.json`**

This file is required by `TilingSpace::LoadFromJson` — without it, the AutoTunerPass
cannot find the search space for the broadcast-add-reduce kernel.

```json
{
  "params": [
    {"name": "TB_M", "min": 16, "max": 128, "step": 16, "alignment": 16},
    {"name": "TB_N", "min": 32, "max": 128, "step": 32, "alignment": 32}
  ],
  "constraints": [
    "TB_M * TB_N * 2 <= ub_size / 2",
    "TB_N <= N"
  ]
}
```

Constraints:
- `TB_M * TB_N * 2 <= ub_size / 2`: UB buffer for one row of B (TB_M × TB_N × 2 bytes f16) fits in half UB; `ub_size` is substituted from HardwareProfile at load time (→ `204800 / 2 = 102400`)
- `TB_N <= N`: tile width cannot exceed actual N dimension

```bash
git add examples/broadcast-add-reduce/tiling_space.json
git commit -m "feat(autotuner): add tiling_space.json for broadcast-add-reduce"
```

- [ ] **Step 3: Build (in xvm)**

```bash
ssh xvm@orb
cd /home/niu/code/Ascend-MLIR
source examples/env.sh
./scripts/build.sh --build-project
```

Common issues:
- Missing include for `mlir::func::FuncOp` → add `#include "mlir/Dialect/Func/IR/FuncOps.h"`
- Missing `arith::MinSIOp` → add `#include "mlir/Dialect/Arith/IR/Arith.h"`
- `emitasc.member` attribute name differs — check actual attr name from step7 IR by running `afir-opt examples/broadcast-add-reduce/step7_kernel.mlir` and inspecting the `emitasc.member` op attributes

- [ ] **Step 4: Run AutoTunerPass on broadcast-add-reduce step7**

```bash
source examples/env.sh
mkdir -p /tmp/autotuner_out
afir-opt examples/broadcast-add-reduce/step7_kernel.mlir \
  --autotuner \
  --autotuner-hw=Ascend910B1 \
  --autotuner-hw-dir=$(pwd)/hardware \
  --autotuner-output-dir=/tmp/autotuner_out \
  --autotuner-space=examples/broadcast-add-reduce/tiling_space.json \
  -o /dev/null
```

Expected: no errors; `/tmp/autotuner_out/` contains `tiling_func_0.cpp` and `get_tiling.cpp`.

- [ ] **Step 5: Inspect generated files**

```bash
cat /tmp/autotuner_out/tiling_func_0.cpp
cat /tmp/autotuner_out/get_tiling.cpp
```

Expected `tiling_func_0.cpp` to contain:
- Function `GetTiling_broadcast_add_reducesum_0` with `int64_t` shape params
- `const int64_t TB_M = <value>;` and `const int64_t TB_N = <value>;`
- `*block_dim_out = ...;`
- `tiling->TB_M = TB_M; tiling->TB_N = TB_N; tiling->dim_arg0_0 = ...; tiling->dim_arg1_1 = ...;`

- [ ] **Step 6: Verify tiling params work with sim-validator**

Extract TB_M and TB_N values from generated `tiling_func_0.cpp`, then verify:

```bash
# Example if TB_M=16, TB_N=32 were chosen:
sim-validator \
  --kernel  examples/broadcast-add-reduce/step8_kernel.cpp \
  --name    broadcast_add_reducesum \
  --tiling-params "TB_M=16,TB_N=32,dim_arg0_0=32,dim_arg1_1=32,dim_arg0_1=32,dim_arg1_0=32" \
  --tiling-layout "int64,int64,int64,int64,int64,int64" \
  --inputs  /tmp/input_a.npy,/tmp/input_b.npy \
  --expected /tmp/expected.npy \
  --block-dim 2 \
  --soc Ascend910B1
```

Note: `broadcast-add-reduce/step8_kernel.cpp` has 6 tiling fields (TB_M, TB_N, dim_arg0_0, dim_arg1_1, dim_arg0_1, dim_arg1_0); adjust `--tiling-params` accordingly. Use the actual TB_M/TB_N values from generated `tiling_func_0.cpp`.

Expected: `PASS`

- [ ] **Step 7: Add lit smoke test**

Create `test/AutoTuner/autotuner_smoke.mlir`:

```mlir
// RUN: afir-opt %s --autotuner --autotuner-hw=Ascend910B1 \
// RUN:   --autotuner-hw-dir=%S/../../hardware \
// RUN:   --autotuner-output-dir=%t \
// RUN:   --autotuner-space=%S/../../examples/broadcast-add-reduce/tiling_space.json \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --allow-empty
// CHECK-NOT: error:
```

The `--autotuner-space` flag points to `examples/broadcast-add-reduce/tiling_space.json` (created in Task 7, Step 1).
Use `examples/broadcast-add-reduce/step7_kernel.mlir` as test input (symlink or copy to `test/AutoTuner/`).

- [ ] **Step 8: Final commit**

```bash
git add test/AutoTuner/
git commit -m "test(autotuner): add smoke test for AutoTunerPass"
```

---

## Acceptance Criteria

- `afir-opt --autotuner` on `broadcast-add-reduce/step7_kernel.mlir` produces `tiling_func_0.cpp` and `get_tiling.cpp` without errors
- Generated `tiling_func_0.cpp` assigns all 4 TilingData fields (`TB_M`, `TB_N`, `dim_arg0_0`, `dim_arg1_1`) and sets `*block_dim_out`
- Tiling params chosen by Solver satisfy the `tiling_space.json` constraints
- `sim-validator` run with generated tiling params produces `PASS` on `broadcast-add-reduce`
