//===- ascend-stage-graph.cpp - MLIR stage graph dumper --------*- C++ -*-===//
//
// Emits a typed operation graph JSON for ascend-debug stage visualization.
//
//===----------------------------------------------------------------------===//

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <string>

using namespace mlir;

static llvm::cl::opt<std::string> InputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input mlir>"), llvm::cl::Required);
static llvm::cl::opt<int> StageOrder("stage-order", llvm::cl::Required,
                                     llvm::cl::desc("Stage order"));
static llvm::cl::opt<std::string> StageName("stage-name",
                                            llvm::cl::Required,
                                            llvm::cl::desc("Stage name"));
static llvm::cl::opt<std::string> StagePath("stage-path",
                                            llvm::cl::Required,
                                            llvm::cl::desc("Stage relative path"));
static llvm::cl::opt<std::string> OutputFilename(
    "output", llvm::cl::Required, llvm::cl::desc("Output graph JSON path"));

namespace {

struct EdgeKey {
  std::string from;
  std::string to;
  std::string value;
  std::string kind;
};

struct EdgeKeyInfo {
  static EdgeKey getEmptyKey() { return {"", "", "", ""}; }
  static EdgeKey getTombstoneKey() { return {"<tombstone>", "", "", ""}; }
  static unsigned getHashValue(const EdgeKey &key) {
    return llvm::hash_combine(key.from, key.to, key.value, key.kind);
  }
  static bool isEqual(const EdgeKey &lhs, const EdgeKey &rhs) {
    return lhs.from == rhs.from && lhs.to == rhs.to && lhs.value == rhs.value &&
           lhs.kind == rhs.kind;
  }
};

class StageGraphBuilder {
public:
  explicit StageGraphBuilder(ModuleOp module) : module(module), asmState(module) {}

  llvm::json::Object build() {
    collectFunctionNames();
    createBlockArgumentNodes();
    createOperationNodes();
    createValueEdges();
    createRegionEdges();

    llvm::json::Object stage;
    stage["order"] = static_cast<int64_t>(StageOrder);
    stage["name"] = StageName;
    stage["path"] = StagePath;

    llvm::json::Array functionsJson;
    for (const std::string &name : functionNames) {
      auto entry = functionNodeIds.find(name);
      if (entry == functionNodeIds.end())
        continue;
      llvm::json::Array ids;
      for (const std::string &id : entry->second)
        ids.push_back(id);
      llvm::json::Object fn;
      fn["name"] = name;
      fn["node_ids"] = std::move(ids);
      functionsJson.push_back(std::move(fn));
    }

    llvm::json::Array kernelIdsJson;
    for (const std::string &id : kernelIds)
      kernelIdsJson.push_back(id);

    llvm::json::Object root;
    root["schema_version"] = 2;
    root["tool"] = "ascend-stage-graph";
    root["graph_source"] = "mlir-tool";
    root["stage"] = std::move(stage);
    root["function"] = functionNames.empty() ? nullptr : functionNames.front();
    root["functions"] = std::move(functionsJson);
    root["nodes"] = std::move(nodes);
    root["edges"] = std::move(edges);
    root["kernel_ids"] = std::move(kernelIdsJson);
    return root;
  }

private:
  ModuleOp module;
  AsmState asmState;
  llvm::json::Array nodes;
  llvm::json::Array edges;
  llvm::DenseMap<Operation *, std::string> opIds;
  llvm::DenseMap<Value, std::string> valueNodeIds;
  llvm::DenseMap<Value, Value> aliasBaseByValue;
  llvm::DenseMap<Value, SmallVector<std::string>> memoryWritersByBase;
  llvm::DenseMap<Value, std::string> resourceWriterByValue;
  llvm::DenseSet<EdgeKey, EdgeKeyInfo> seenEdges;
  llvm::SmallVector<std::string> functionNames;
  llvm::StringMap<SmallVector<std::string>> functionNodeIds;
  llvm::SmallVector<std::string> kernelIds;
  std::string lastEffectNode;
  std::string pendingBarrierNode;

  static bool isSkippedOperation(Operation *op) {
    return isa<ModuleOp>(op) || isa<func::FuncOp>(op);
  }

  std::string nextNodeId() { return "n" + std::to_string(nodes.size()); }

  std::string nextEdgeId() { return "e" + std::to_string(edges.size()); }

  std::string printValue(Value value) {
    std::string storage;
    llvm::raw_string_ostream os(storage);
    value.print(os, asmState);
    return storage;
  }

  std::string printType(Type type) {
    std::string storage;
    llvm::raw_string_ostream os(storage);
    type.print(os);
    return storage;
  }

  std::string printOperation(Operation *op) {
    std::string storage;
    llvm::raw_string_ostream os(storage);
    op->print(os, asmState);
    return storage;
  }

  std::optional<unsigned> lineFromLoc(Location loc) {
    if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
      return fileLoc.getLine();
    if (auto nameLoc = dyn_cast<NameLoc>(loc))
      return lineFromLoc(nameLoc.getChildLoc());
    if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
      for (Location child : fusedLoc.getLocations())
        if (auto line = lineFromLoc(child))
          return line;
    }
    if (auto callLoc = dyn_cast<CallSiteLoc>(loc))
      return lineFromLoc(callLoc.getCallee());
    return std::nullopt;
  }

  std::string currentFunctionName(Operation *op) {
    if (auto func = dyn_cast<func::FuncOp>(op))
      return func.getName().str();
    if (auto func = op->getParentOfType<func::FuncOp>())
      return func.getName().str();
    return "";
  }

  Value aliasBase(Value value) {
    llvm::DenseSet<Value> seen;
    Value current = value;
    while (aliasBaseByValue.count(current) && !seen.contains(current)) {
      seen.insert(current);
      current = aliasBaseByValue[current];
    }
    return current;
  }

  llvm::json::Array valuesToJson(ValueRange values) {
    llvm::json::Array array;
    for (Value value : values)
      array.push_back(printValue(value));
    return array;
  }

  llvm::json::Array resultTypesToJson(ResultRange values) {
    llvm::json::Array array;
    for (Value value : values)
      array.push_back(printType(value.getType()));
    return array;
  }

  std::string resultTypeSummary(Operation *op) {
    if (op->getNumResults() == 0)
      return "";
    std::string storage;
    llvm::raw_string_ostream os(storage);
    llvm::interleaveComma(op->getResultTypes(), os,
                          [&](Type type) { type.print(os); });
    return storage;
  }

  void collectFunctionNames() {
    module.walk([&](func::FuncOp func) {
      functionNames.push_back(func.getName().str());
      functionNodeIds[func.getName()].clear();
    });
  }

  void addNodeIdToFunction(StringRef function, const std::string &id) {
    if (!function.empty())
      functionNodeIds[function].push_back(id);
  }

  void createBlockArgumentNode(BlockArgument arg, StringRef function) {
    std::string id = nextNodeId();
    valueNodeIds[arg] = id;

    llvm::json::Object node;
    node["id"] = id;
    node["line"] = 0;
    node["line_end"] = 0;
    node["function"] = function.str();
    Operation *parentOp = arg.getOwner()->getParentOp();
    bool functionArg = isa_and_nonnull<func::FuncOp>(parentOp);
    node["op_name"] = functionArg ? "func.arg" : "block.arg";
    node["label"] = printValue(arg);
    node["input_values"] = llvm::json::Array();
    llvm::json::Array results;
    results.push_back(printValue(arg));
    node["result_values"] = std::move(results);
    node["result_type"] = printType(arg.getType());
    node["source_excerpt"] = "";
    node["region_body"] = nullptr;
    node["body_ops"] = llvm::json::Array();
    node["body_summary"] = nullptr;
    node["kernel_id"] = nullptr;
    node["op_role"] = nullptr;
    node["schedule_decision_id"] = nullptr;
    node["workspace_size_bytes"] = nullptr;
    node["semantic_attrs"] = llvm::json::Object();
    node["badges"] = llvm::json::Array();
    nodes.push_back(std::move(node));
    addNodeIdToFunction(function, id);
  }

  void createBlockArgumentNodes() {
    module.walk([&](Operation *op) {
      if (!isa<func::FuncOp, scf::ForOp, scf::IfOp>(op) && op != module)
        return;
      std::string function = currentFunctionName(op);
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments()) {
            if (!valueNodeIds.count(arg))
              createBlockArgumentNode(arg, function);
          }
        }
      }
    });
  }

  void createOperationNodes() {
    module.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isSkippedOperation(op))
        return;
      std::string id = nextNodeId();
      opIds[op] = id;
      for (Value result : op->getResults())
        valueNodeIds[result] = id;

      std::string function = currentFunctionName(op);
      std::string opName = op->getName().getStringRef().str();
      std::optional<unsigned> line = lineFromLoc(op->getLoc());

      llvm::json::Object node;
      node["id"] = id;
      node["line"] = line ? *line : 0;
      node["line_end"] = line ? *line : 0;
      node["function"] = function.str();
      node["op_name"] = opName;
      node["label"] = op->getNumResults() ? printValue(op->getResult(0)) : opName;
      node["input_values"] = valuesToJson(op->getOperands());
      llvm::json::Array resultValues;
      for (Value result : op->getResults())
        resultValues.push_back(printValue(result));
      node["result_values"] = std::move(resultValues);
      std::string resultType = resultTypeSummary(op);
      node["result_type"] = resultType.empty() ? nullptr : llvm::json::Value(resultType);
      node["source_excerpt"] = printOperation(op);
      node["region_body"] = nullptr;
      node["body_ops"] = llvm::json::Array();
      node["body_summary"] = nullptr;

      if (auto attr = op->getAttrOfType<StringAttr>("ascend.kernel")) {
        node["kernel_id"] = attr.getValue().str();
        if (!llvm::is_contained(kernelIds, attr.getValue().str()))
          kernelIds.push_back(attr.getValue().str());
      } else {
        node["kernel_id"] = nullptr;
      }
      if (auto attr = op->getAttrOfType<StringAttr>("ascend.op_role"))
        node["op_role"] = attr.getValue().str();
      else
        node["op_role"] = nullptr;
      if (auto attr =
              op->getAttrOfType<StringAttr>("ascend.schedule.decision_id"))
        node["schedule_decision_id"] = attr.getValue().str();
      else
        node["schedule_decision_id"] = nullptr;
      if (auto attr = op->getAttrOfType<IntegerAttr>("cann.workspace_size_bytes"))
        node["workspace_size_bytes"] = attr.getInt();
      else
        node["workspace_size_bytes"] = nullptr;
      node["semantic_attrs"] = llvm::json::Object();
      node["badges"] = llvm::json::Array();
      nodes.push_back(std::move(node));
      addNodeIdToFunction(function, id);
    });
  }

  void appendEdge(const std::string &from, const std::string &to,
                  const std::string &value, StringRef kind,
                  StringRef effect = "", StringRef label = "") {
    if (from.empty() || to.empty() || from == to)
      return;
    EdgeKey key{from, to, value, kind.str()};
    if (seenEdges.contains(key))
      return;
    seenEdges.insert(key);
    llvm::json::Object edge;
    edge["id"] = nextEdgeId();
    edge["from"] = from;
    edge["to"] = to;
    edge["value"] = value;
    edge["kind"] = kind.str();
    if (!effect.empty())
      edge["effect"] = effect.str();
    if (!label.empty())
      edge["label"] = label.str();
    edges.push_back(std::move(edge));
  }

  bool isMemoryObserver(StringRef opName) {
    return opName == "func.return" || opName == "memref.copy" ||
           opName == "memref.load" || opName == "memref.store" ||
           opName == "affine.load" || opName == "affine.store" ||
           opName == "vector.transfer_read" ||
           opName == "vector.transfer_write" ||
           opName.starts_with("linalg.");
  }

  SmallVector<Value> memoryWrittenBases(Operation *op) {
    StringRef opName = op->getName().getStringRef();
    SmallVector<Value> written;
    if (opName == "memref.copy" && op->getNumOperands() >= 2)
      written.push_back(aliasBase(op->getOperand(1)));
    return written;
  }

  bool isTensorLike(Value value) {
    std::string type = printType(value.getType());
    return llvm::StringRef(type).contains("tensor") ||
           llvm::StringRef(type).contains("Tensor");
  }

  void classifyResources(Operation *op, SmallVectorImpl<Value> &reads,
                         SmallVectorImpl<Value> &writes) {
    StringRef opName = op->getName().getStringRef();
    if (opName == "ascendc.pipe.init_queue" && op->getNumOperands() >= 2) {
      reads.push_back(op->getOperand(0));
      writes.push_back(op->getOperand(1));
      return;
    }
    if (opName == "ascendc.pipe.init_buffer" && op->getNumOperands() >= 2) {
      reads.push_back(op->getOperand(0));
      writes.push_back(op->getOperand(1));
      return;
    }
    if (opName == "ascendc.que_bind.alloc_tensor" && op->getNumOperands() >= 1) {
      reads.push_back(op->getOperand(0));
      writes.push_back(op->getOperand(0));
      return;
    }
    if (opName == "ascendc.que_bind.enque_tensor" && op->getNumOperands() >= 2) {
      reads.push_back(op->getOperand(1));
      writes.push_back(op->getOperand(0));
      return;
    }
    if (opName == "ascendc.que_bind.deque_tensor" && op->getNumOperands() >= 1) {
      reads.push_back(op->getOperand(0));
      writes.push_back(op->getOperand(0));
      return;
    }
    if (opName == "ascendc.que_bind.free_tensor" && op->getNumOperands() >= 2) {
      reads.push_back(op->getOperand(1));
      writes.push_back(op->getOperand(0));
      return;
    }
    if (opName == "ascendc.global_tensor.set_global_buffer" &&
        op->getNumOperands() >= 1) {
      writes.push_back(op->getOperand(0));
      for (Value operand : op->getOperands().drop_front())
        reads.push_back(operand);
      return;
    }
    if ((opName.starts_with("ascendc.") || opName == "emitasc.verbatim") &&
        op->getNumOperands() > 0) {
      if (isTensorLike(op->getOperand(0)))
        writes.push_back(op->getOperand(0));
      for (Value operand : op->getOperands().drop_front()) {
        if (isTensorLike(operand))
          reads.push_back(operand);
      }
    }
  }

  void connectPriorResourceWriters(Value resource, const std::string &nodeId,
                                   StringRef effect) {
    Value base = aliasBase(resource);
    auto it = resourceWriterByValue.find(base);
    if (it == resourceWriterByValue.end())
      return;
    std::string valueName = printValue(base);
    appendEdge(it->second, nodeId, valueName, "resource_effect", effect,
               (effect + " " + valueName).str());
  }

  void recordResourceWriter(Value resource, const std::string &nodeId) {
    Value base = aliasBase(resource);
    resourceWriterByValue[base] = nodeId;
  }

  void createValueEdges() {
    module.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isSkippedOperation(op))
        return;
      std::string nodeId = opIds.lookup(op);
      if (nodeId.empty())
        return;
      StringRef opName = op->getName().getStringRef();

      for (Value operand : op->getOperands()) {
        auto it = valueNodeIds.find(operand);
        if (it != valueNodeIds.end())
          appendEdge(it->second, nodeId, printValue(operand), "value");
      }

      SmallVector<Value> memoryBases;
      for (Value operand : op->getOperands()) {
        Value base = aliasBase(operand);
        if (!llvm::is_contained(memoryBases, base))
          memoryBases.push_back(base);
      }
      if (isMemoryObserver(opName)) {
        for (Value base : memoryBases) {
          auto it = memoryWritersByBase.find(base);
          if (it == memoryWritersByBase.end())
            continue;
          std::string valueName = printValue(base);
          for (const std::string &writer : it->second)
            appendEdge(writer, nodeId, valueName, "memory_effect", "write",
                       "write " + valueName);
        }
      }

      SmallVector<Value> resourceReads;
      SmallVector<Value> resourceWrites;
      classifyResources(op, resourceReads, resourceWrites);
      bool isEffectOp = !resourceReads.empty() || !resourceWrites.empty();
      if (opName == "ascendc.pipe_barrier") {
        if (!lastEffectNode.empty())
          appendEdge(lastEffectNode, nodeId, "pipe_all", "control", "barrier",
                     "barrier pipe_all");
        pendingBarrierNode = nodeId;
        lastEffectNode = nodeId;
        return;
      }

      if (isEffectOp && !pendingBarrierNode.empty()) {
        appendEdge(pendingBarrierNode, nodeId, "pipe_all", "control", "barrier",
                   "barrier pipe_all");
        pendingBarrierNode.clear();
      }
      for (Value resource : resourceReads)
        connectPriorResourceWriters(resource, nodeId, "read");
      for (Value resource : resourceWrites)
        connectPriorResourceWriters(resource, nodeId, "write");

      for (Value result : op->getResults()) {
        if (opName == "memref.subview" || opName == "memref.cast" ||
            opName == "memref.reinterpret_cast") {
          if (op->getNumOperands() >= 1)
            aliasBaseByValue[result] = aliasBase(op->getOperand(0));
        }
      }
      for (Value base : memoryWrittenBases(op)) {
        auto &writers = memoryWritersByBase[base];
        if (!llvm::is_contained(writers, nodeId))
          writers.push_back(nodeId);
      }
      for (Value resource : resourceWrites)
        recordResourceWriter(resource, nodeId);
      if (isEffectOp)
        lastEffectNode = nodeId;
    });
  }

  void createRegionEdges() {
    for (const auto &entry : opIds) {
      Operation *op = entry.first;
      std::string nodeId = entry.second;
      Operation *parent = op->getParentOp();
      while (parent && !opIds.count(parent))
        parent = parent->getParentOp();
      if (parent)
        appendEdge(opIds.lookup(parent), nodeId, "region", "region");
    }

    for (const auto &entry : valueNodeIds) {
      Value value = entry.first;
      auto arg = dyn_cast<BlockArgument>(value);
      if (!arg)
        continue;
      Operation *parent = arg.getOwner()->getParentOp();
      while (parent && !opIds.count(parent))
        parent = parent->getParentOp();
      if (parent)
        appendEdge(opIds.lookup(parent), entry.second, printValue(arg), "region");
    }
  }
};

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Ascend MLIR stage graph dumper\n");

  DialectRegistry registry;
  registry.insert<affine::AffineDialect, arith::ArithDialect,
                  cf::ControlFlowDialect, func::FuncDialect,
                  linalg::LinalgDialect, math::MathDialect,
                  memref::MemRefDialect, scf::SCFDialect,
                  tensor::TensorDialect, vector::VectorDialect,
                  ascendc::AscendCDialect, emitasc::EmitAscDialect>();
  ascendc::registerExternalModels(registry);
  emitasc::registerExternalModels(registry);

  MLIRContext context(registry);
  context.allowUnregisteredDialects();

  auto bufferOrError = llvm::MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (!bufferOrError) {
    llvm::errs() << "cannot open input: " << bufferOrError.getError().message()
                 << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*bufferOrError), llvm::SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "failed to parse input MLIR\n";
    return 1;
  }

  StageGraphBuilder builder(*module);
  llvm::json::Object graph = builder.build();

  std::error_code error;
  llvm::ToolOutputFile output(OutputFilename, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << "cannot open output: " << error.message() << "\n";
    return 1;
  }
  llvm::json::OStream json(output.os(), /*IndentSize=*/2);
  json.value(llvm::json::Value(std::move(graph)));
  output.keep();
  return 0;
}
