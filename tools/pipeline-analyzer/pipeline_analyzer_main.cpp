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
    if (c == '"')       out << "\\\"";
    else if (c == '\\') out << "\\\\";
    else                out << c;
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
        if (lbl.rfind(prefix + ".", 0) == 0)
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
  out << "{\n\"displayTimeUnit\": \"ns\",\n\"traceEvents\": [\n";
  bool first = true;
  for (auto& e : all_events) emitEvent(out, e, first);
  out << "\n]\n}\n";
  out.close();

  llvm::outs() << "Wrote: " << OutputFile << "  (" << all_events.size() << " events)\n";
  llvm::outs() << "Open in Chrome: chrome://tracing → Load → " << OutputFile << "\n";

  llvm::outs().flush();
  return 0;
}
