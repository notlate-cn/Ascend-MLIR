# Pipeline Analyzer CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `pipeline-analyzer` CLI tool that reads AscendC simulator dump files (`core*.veccore*.ccu.*_issque.dump`), produces per-unit cycle statistics, and emits a Chrome Trace Format JSON file viewable in `chrome://tracing`.

**Architecture:** Single-file CLI `tools/pipeline-analyzer/pipeline_analyzer_main.cpp`. No runtime library dependency (reads plain text dump files, emits JSON). Parses `push_isa`/`retire_isa` entries from each unit's issue-queue dump, computes `dur = retire_cycle - push_cycle` per instruction, builds Chrome Trace events (one lane per unit per core), writes `.json` output. Independent of SimValidator/autotuner — can be used standalone after any simulator run.

**Tech Stack:** C++17, LLVM Support (CommandLine, FileSystem, raw_ostream, JSON or manual string emit). No new libraries.

---

## Simulator Dump File Reference

The simulator writes these files in the run directory (e.g. `sim/`) when `ASCEND_CPU_SIMULATION=1`:

| File pattern | Pipeline unit | pipe id |
|---|---|---|
| `core{C}.veccore{V}.ccu.mte1_issque.dump` | MTE1 (UB→L1, rarely used) | 3 |
| `core{C}.veccore{V}.ccu.mte2_issque.dump` | MTE2 (GM→UB load) | 4 |
| `core{C}.veccore{V}.ccu.mte3_issque.dump` | MTE3 (UB→GM store) | 5 |
| `core{C}.veccore{V}.ccu.vec_issque.dump` | VEC (vector compute) | 1 |
| `core{C}.veccore{V}.ccu.scalar_issque.dump` | SCALAR | 2 |
| `core{C}_summary_log` | Total ticks per core | — |
| `core{C}.cubecore{V}.ccu.*_issque.dump` | Cube core units (parsed same as veccore) | — |

Each `*_issque.dump` line format:
```
[info] <cycle>:  pipe: <N> push_isa, pc:0x<hex>, id:<instr_id>
[info] <cycle>:  pipe: <N> retire_isa, pc:0x<hex>, id:<instr_id>
[debug] <cycle>: pipe: <N> can_not_releae_wait_flag, pc:0x<hex>.
```

Chrome Trace Format (output):
```json
{"traceEvents": [
  {"name": "MTE2", "ph": "X", "ts": 1419, "dur": 386, "pid": 0, "tid": 1,
   "args": {"pc": "0x10d111f4", "id": 125, "core": "core0.veccore0"}},
  ...
], "displayTimeUnit": "ns"}
```
- `ts` = push cycle, `dur` = retire_cycle - push_cycle
- `pid` = veccore index (each `core{C}.veccore{V}` gets its own pid/row in viewer, sorted alphabetically)
- `tid` = unit index (MTE1=0, MTE2=1, MTE3=2, VEC=3, SCALAR=4, STALL=5)
- STALL events: gaps between consecutive instructions in same unit shown as synthetic "stall" events

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `tools/pipeline-analyzer/pipeline_analyzer_main.cpp` | Create | Full CLI: parse dumps, emit Chrome Trace JSON |
| `tools/pipeline-analyzer/CMakeLists.txt` | Create | Build rules |
| `CMakeLists.txt` (root) | Modify | Add `add_subdirectory(tools/pipeline-analyzer)` |

---

## Task 1: pipeline-analyzer CLI

**Files:**
- Create: `tools/pipeline-analyzer/CMakeLists.txt`
- Create: `tools/pipeline-analyzer/pipeline_analyzer_main.cpp`
- Modify: `CMakeLists.txt` (root)

- [ ] **Step 1: Write `tools/pipeline-analyzer/CMakeLists.txt`**

```cmake
# tools/pipeline-analyzer/CMakeLists.txt
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(pipeline-analyzer
  pipeline_analyzer_main.cpp
)

target_include_directories(pipeline-analyzer PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
```

- [ ] **Step 2: Add to root `CMakeLists.txt`**

After `add_subdirectory(tools/sim-validator)`:
```cmake
add_subdirectory(tools/pipeline-analyzer)
```

- [ ] **Step 3: Write `tools/pipeline-analyzer/pipeline_analyzer_main.cpp`**

Write the full implementation:

```cpp
// tools/pipeline-analyzer/pipeline_analyzer_main.cpp
//
// Reads AscendC simulator issue-queue dumps and emits Chrome Trace JSON.
// Usage: pipeline-analyzer --sim-dir sim/ --output trace.json
// View:  chrome://tracing → Load → trace.json

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace llvm;

static cl::opt<std::string> SimDir("sim-dir",
    cl::desc("Simulator output directory containing dump files"),
    cl::init("."));
static cl::opt<std::string> OutputFile("output",
    cl::desc("Output Chrome Trace JSON file"), cl::init("trace.json"));
static cl::opt<bool> ShowStalls("stalls",
    cl::desc("Emit synthetic stall events for gaps between instructions"),
    cl::init(false));

// ── Data structures ──────────────────────────────────────────────────────────

struct TraceEvent {
  std::string name;   // unit name: MTE1/MTE2/MTE3/VEC/SCALAR
  int64_t     ts;     // push cycle
  int64_t     dur;    // retire - push
  int         pid;    // core index
  int         tid;    // unit index (lane in viewer)
  std::string pc;
  int64_t     instr_id;
  std::string core_label; // e.g. "core0.veccore1"
};

// unit name → tid lane assignment
static int unitTid(const std::string& unit) {
  if (unit == "MTE1")   return 0;
  if (unit == "MTE2")   return 1;
  if (unit == "MTE3")   return 2;
  if (unit == "VEC")    return 3;
  if (unit == "SCALAR") return 4;
  if (unit == "STALL")  return 5;
  return 6;
}

// Map dump file suffix → unit name
static std::string dumpSuffix2Unit(const std::string& filename) {
  if (filename.find("mte1_issque") != std::string::npos) return "MTE1";
  if (filename.find("mte2_issque") != std::string::npos) return "MTE2";
  if (filename.find("mte3_issque") != std::string::npos) return "MTE3";
  if (filename.find("vec_issque")  != std::string::npos) return "VEC";
  if (filename.find("scalar_issque") != std::string::npos) return "SCALAR";
  return "";
}

// ── Parser ───────────────────────────────────────────────────────────────────

// Parse one *_issque.dump file. Returns list of complete (push+retire) events.
// Incomplete pairs (push without retire) are skipped.
static std::vector<TraceEvent> parseDump(
    const std::string& path,
    const std::string& unit,
    int pid,
    const std::string& core_label) {

  auto buf = MemoryBuffer::getFile(path);
  if (!buf) return {};

  // id → push_cycle
  std::map<int64_t, int64_t> pending;
  // id → pc
  std::map<int64_t, std::string> pending_pc;

  std::vector<TraceEvent> events;

  SmallVector<StringRef> lines;
  (*buf)->getBuffer().split(lines, '\n');

  for (auto line : lines) {
    line = line.trim();
    if (line.empty()) continue;

    // Format: "[info] <cycle>:  pipe: <N> push_isa, pc:0x<hex>, id:<N>"
    //         "[info] <cycle>:  pipe: <N> retire_isa, pc:0x<hex>, id:<N>"

    bool is_push   = line.contains("push_isa");
    bool is_retire = line.contains("retire_isa");
    if (!is_push && !is_retire) continue;

    // Extract cycle: between "] " and ":"
    auto bracket = line.find("] ");
    auto colon1  = line.find(':', bracket == StringRef::npos ? 0 : bracket);
    if (bracket == StringRef::npos || colon1 == StringRef::npos) continue;
    int64_t cycle = 0;
    StringRef cycle_str = line.substr(bracket + 2, colon1 - bracket - 2).trim();
    if (cycle_str.getAsInteger(10, cycle)) continue;

    // Extract id: "id:<N>"
    auto id_pos = line.find("id:");
    if (id_pos == StringRef::npos) continue;
    int64_t instr_id = 0;
    if (line.substr(id_pos + 3).trim().getAsInteger(10, instr_id)) continue;

    // Extract pc: "pc:0x<hex>"
    std::string pc;
    auto pc_pos = line.find("pc:");
    if (pc_pos != StringRef::npos) {
      auto comma = line.find(',', pc_pos);
      pc = line.substr(pc_pos + 3,
                       comma == StringRef::npos ? StringRef::npos : comma - pc_pos - 3)
               .trim().str();
    }

    if (is_push) {
      pending[instr_id]    = cycle;
      pending_pc[instr_id] = pc;
    } else { // retire
      auto it = pending.find(instr_id);
      if (it == pending.end()) continue; // no matching push
      int64_t push_cycle = it->second;
      int64_t dur = cycle - push_cycle;
      if (dur < 0) dur = 0;
      TraceEvent ev;
      ev.name       = unit;
      ev.ts         = push_cycle;
      ev.dur        = dur;
      ev.pid        = pid;
      ev.tid        = unitTid(unit);
      ev.pc         = pending_pc[instr_id];
      ev.instr_id   = instr_id;
      ev.core_label = core_label;
      events.push_back(ev);
      pending.erase(it);
      pending_pc.erase(instr_id);
    }
  }
  return events;
}

// ── Stats ────────────────────────────────────────────────────────────────────

struct UnitStats {
  std::string unit;
  int64_t     total_active = 0;  // sum of dur
  int64_t     total_stall  = 0;  // gaps
  int64_t     count        = 0;
  double      efficiency   = 0.0; // active / (active + stall)
};

static UnitStats computeStats(const std::vector<TraceEvent>& events,
                               const std::string& unit, int64_t total_cycles) {
  UnitStats s;
  s.unit = unit;
  for (auto& e : events) if (e.name == unit) { s.total_active += e.dur; ++s.count; }
  s.total_stall = total_cycles > 0 ? total_cycles - s.total_active : 0;
  if (s.total_stall < 0) s.total_stall = 0;
  s.efficiency = total_cycles > 0 ? static_cast<double>(s.total_active) / total_cycles : 0.0;
  return s;
}

// ── Stall events ─────────────────────────────────────────────────────────────

static std::vector<TraceEvent> makeStallEvents(
    std::vector<TraceEvent> unit_events, int pid, const std::string& core_label) {
  if (unit_events.empty()) return {};
  std::sort(unit_events.begin(), unit_events.end(),
            [](const TraceEvent& a, const TraceEvent& b){ return a.ts < b.ts; });

  std::vector<TraceEvent> stalls;
  int64_t prev_end = unit_events[0].ts;
  std::string unit = unit_events[0].name;
  for (auto& e : unit_events) {
    if (e.ts > prev_end) {
      TraceEvent stall;
      stall.name       = "STALL";
      stall.ts         = prev_end;
      stall.dur        = e.ts - prev_end;
      stall.pid        = pid;
      stall.tid        = unitTid("STALL");
      stall.pc         = "";
      stall.instr_id   = -1;
      stall.core_label = core_label;
      stalls.push_back(stall);
    }
    prev_end = e.ts + e.dur;
  }
  return stalls;
}

// ── JSON emit ─────────────────────────────────────────────────────────────────

static void jsonString(std::ostream& out, const std::string& s) {
  out << '"';
  for (char c : s) {
    if (c == '"')  out << "\\\"";
    else if (c == '\\') out << "\\\\";
    else           out << c;
  }
  out << '"';
}

static void emitEvent(std::ostream& out, const TraceEvent& e, bool& first) {
  if (!first) out << ",\n";
  first = false;
  out << "  {\"name\":";   jsonString(out, e.name);
  out << ",\"ph\":\"X\"";
  out << ",\"ts\":"  << e.ts;
  out << ",\"dur\":" << (e.dur > 0 ? e.dur : 1);
  out << ",\"pid\":" << e.pid;
  out << ",\"tid\":" << e.tid;
  out << ",\"args\":{\"core\":";  jsonString(out, e.core_label);
  if (!e.pc.empty()) { out << ",\"pc\":"; jsonString(out, e.pc); }
  if (e.instr_id >= 0) out << ",\"id\":" << e.instr_id;
  out << "}}";
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv, "AscendC Pipeline Analyzer\n");

  std::string sim_dir = SimDir;
  // Resolve to absolute path
  {
    llvm::SmallString<256> abs(sim_dir);
    llvm::sys::fs::make_absolute(abs);
    sim_dir = abs.str().str();
  }

  // Discover dump files: core{C}.veccore{V}.ccu.*_issque.dump
  // Build map: core_label → list of (unit, path)
  struct DumpFile { std::string unit; std::string path; };
  std::map<std::string, std::vector<DumpFile>> core_dumps;  // core_label → dumps

  std::error_code ec;
  for (llvm::sys::fs::directory_iterator it(sim_dir, ec), end;
       !ec && it != end; it.increment(ec)) {
    std::string fname = llvm::sys::path::filename(it->path()).str();
    std::string unit  = dumpSuffix2Unit(fname);
    if (unit.empty()) continue;

    // Extract core_label: everything before ".ccu."
    auto ccu = fname.find(".ccu.");
    if (ccu == std::string::npos) continue;
    std::string core_label = fname.substr(0, ccu);
    core_dumps[core_label].push_back({unit, it->path()});
  }

  if (core_dumps.empty()) {
    llvm::errs() << "No *_issque.dump files found in: " << sim_dir << "\n";
    return 1;
  }

  // Parse summary logs for total cycle counts
  std::map<std::string, int64_t> core_total_cycles;
  for (llvm::sys::fs::directory_iterator it(sim_dir, ec), end;
       !ec && it != end; it.increment(ec)) {
    std::string fname = llvm::sys::path::filename(it->path()).str();
    if (fname.find("_summary_log") == std::string::npos) continue;
    // core label from summary: "core0_summary_log" → "core0"
    // but we need veccore granularity — use system ticks from summary
    // Map: summary prefix "core0" applies to all core0.veccore* entries
    std::string prefix = fname.substr(0, fname.find("_summary_log"));
    auto buf = MemoryBuffer::getFile(it->path());
    if (!buf) continue;
    SmallVector<StringRef> lines;
    (*buf)->getBuffer().split(lines, '\n');
    for (auto line : lines) {
      if (!line.contains("kernal total ticks")) continue;
      auto c = line.rfind(':');
      if (c == StringRef::npos) continue;
      int64_t ticks = 0;
      if (line.substr(c + 1).trim().getAsInteger(10, ticks)) continue;
      // Associate with all veccores under this core
      for (auto& [lbl, _] : core_dumps)
        if (lbl.starts_with(prefix + "."))
          if (ticks > core_total_cycles[lbl]) core_total_cycles[lbl] = ticks;
    }
  }

  // Collect all events
  std::vector<TraceEvent> all_events;
  int pid = 0;
  std::vector<std::string> sorted_cores;
  for (auto& [lbl, _] : core_dumps) sorted_cores.push_back(lbl);
  std::sort(sorted_cores.begin(), sorted_cores.end());

  llvm::outs() << "Cores found: " << sorted_cores.size() << "\n";

  for (auto& core_label : sorted_cores) {
    auto& dumps = core_dumps[core_label];
    int64_t total = core_total_cycles.count(core_label)
                    ? core_total_cycles[core_label] : 0;

    llvm::outs() << "  " << core_label
                 << "  total_cycles=" << total << "\n";

    std::map<std::string, std::vector<TraceEvent>> unit_events;

    for (auto& df : dumps) {
      auto evs = parseDump(df.path, df.unit, pid, core_label);
      for (auto& e : evs) unit_events[df.unit].push_back(e);
      all_events.insert(all_events.end(), evs.begin(), evs.end());

      // Print stats
      UnitStats st = computeStats(evs, df.unit, total);
      llvm::outs() << "    " << df.unit
                   << "  instructions=" << st.count
                   << "  active_cycles=" << st.total_active
                   << "  efficiency=" << (int)(st.efficiency * 100) << "%\n";
    }

    // Stall events
    if (ShowStalls) {
      for (auto& [unit, evs] : unit_events) {
        auto stalls = makeStallEvents(evs, pid, core_label);
        all_events.insert(all_events.end(), stalls.begin(), stalls.end());
      }
    }

    ++pid;
  }

  // Emit Chrome Trace JSON
  std::ofstream out(OutputFile);
  if (!out) {
    llvm::errs() << "Error: cannot write to " << OutputFile << "\n";
    return 1;
  }

  // Note: ts/dur are simulator cycle counts, not nanoseconds.
  // displayTimeUnit is informational only in chrome://tracing Gantt view.
  out << "{\n\"displayTimeUnit\": \"us\",\n\"traceEvents\": [\n";
  bool first = true;
  for (auto& e : all_events) emitEvent(out, e, first);
  out << "\n]\n}\n";
  out.close();

  llvm::outs() << "Wrote: " << OutputFile << "  (" << all_events.size() << " events)\n";
  llvm::outs() << "Open in Chrome: chrome://tracing → Load → " << OutputFile << "\n";

  llvm::outs().flush();
  return 0;
}
```

- [ ] **Step 4: Build**

```bash
# In xvm:
sleep 1
cd /home/niu/code/Ascend-MLIR
./scripts/build.sh --build-project --llvm-build-dir ~/code/llvm-project/build 2>&1 | tail -10
```
Expected: `pipeline-analyzer` binary in `build/bin/`.

- [ ] **Step 5: Commit**

```bash
git add tools/pipeline-analyzer/ CMakeLists.txt
git commit -m "feat(pipeline-analyzer): add Chrome Trace JSON generator for sim dumps"
```

---

## Task 2: Integration test

- [ ] **Step 1: Run a sim-validator pass to generate fresh dumps**

```bash
# In xvm sim/ dir with env set:
cd /home/niu/code/Ascend-MLIR/sim
sim-validator \
  --kernel   ../examples/broadcast-add-reduce/step8_kernel-adjust.cpp \
  --name     broadcast_add_reducesum \
  --tiling-params "TB_M=16,TB_N=16,dim_arg0_0=32,dim_arg1_1=32,dim_arg0_1=32,dim_arg1_0=32" \
  --tiling-layout "int64,int64,int64,int64,int64,int64" \
  --inputs   /tmp/input_a.npy,/tmp/input_b.npy \
  --expected /tmp/expected.npy \
  --block-dim 2
```

- [ ] **Step 2: Run pipeline-analyzer**

```bash
cd /home/niu/code/Ascend-MLIR/sim
pipeline-analyzer --sim-dir . --output /tmp/trace.json --stalls
```

Expected output (approximate — actual core count depends on `--block-dim` used in step 1):
```
Cores found: 4
  core0.cubecore0  total_cycles=0
    ...
  core0.veccore0  total_cycles=11416
    MTE2  instructions=<N>  active_cycles=<N>  efficiency=<N>%
    VEC   instructions=<N>  active_cycles=<N>  efficiency=<N>%
    MTE3  instructions=<N>  active_cycles=<N>  efficiency=<N>%
    ...
  core0.veccore1  total_cycles=11416
    ...
  core1.veccore0  total_cycles=0
    ...
Wrote: /tmp/trace.json  (<N> events)
Open in Chrome: chrome://tracing → Load → /tmp/trace.json
```

Notes:
- `cubecore` entries appear because `cubecore*.ccu.mte2_issque.dump` files exist in the sim dir — they are parsed correctly and appear as separate rows in chrome://tracing.
- Cores without a `*_summary_log` (e.g. core1 when only `core0_summary_log` exists) will show `total_cycles=0` and `efficiency=0%`. This is correct behavior — efficiency computation requires total cycle count from summary logs.
- Both `core0.veccore0` and `core0.veccore1` get the same `total_cycles` value (max across all `kernal total ticks` entries in `core0_summary_log`).

- [ ] **Step 3: Verify JSON is valid and loadable**

```bash
python3 -c "import json; d=json.load(open('/tmp/trace.json')); print(len(d['traceEvents']), 'events')"
```
Expected: no error, event count matches.

Copy `/tmp/trace.json` to Mac and open in `chrome://tracing` to verify Gantt visualization.

- [ ] **Step 4: Commit test note**

```bash
git commit --allow-empty -m "test(pipeline-analyzer): integration test PASS, trace.json valid"
```

---

## Acceptance Criteria

- `pipeline-analyzer --help` runs without error
- Running on `sim/` after a sim-validator pass produces valid `trace.json`
- JSON loads in `chrome://tracing` without error
- Multiple cores shown as separate `pid` rows
- MTE2/VEC/MTE3/SCALAR shown as separate `tid` lanes per core
- With `--stalls`, gap events appear between instructions in same lane
- `python3 -c "import json; json.load(open('trace.json'))"` succeeds
