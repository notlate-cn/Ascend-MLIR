//===- AFIRToASCIRText.cpp - AFIR to ASCIR text conversion ----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AFIRToASCIRText/AFIRToASCIRText.h"
#include "Dialect/AFIR/AFIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

#define GEN_PASS_DECL_CONVERTAFIRTOASCIRTEXTPASS
#define GEN_PASS_DEF_CONVERTAFIRTOASCIRTEXTPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir {
namespace afir {

struct AFIRToASCIRTextPass : public ::impl::ConvertAFIRToASCIRTextPassBase<AFIRToASCIRTextPass> {
  using Base = ::impl::ConvertAFIRToASCIRTextPassBase<AFIRToASCIRTextPass>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    std::string output = convertModuleToASCIRText(module);

    if (!ascirPath.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream os(ascirPath, ec);
      if (ec) {
        llvm::errs() << "Error opening file '" << ascirPath << "': " << ec.message() << "\n";
        return signalPassFailure();
      }
      os << output;
      os.close();
    } else {
      llvm::errs() << output;
    }
  }

 private:
  struct OpInfo {
    int apiType;
    int apiUnit;
    int computeType;
    bool isBroadcast;
  };

  OpInfo getOpInfo(StringRef opName);
  std::string convertModuleToASCIRText(ModuleOp module);
  std::string convertFuncToASCIRText(func::FuncOp func, Attribute graphAttr);
  std::string convertOpToASCIRText(Operation *op, int &nodeCounter, llvm::StringMap<std::string> &ssaToNodeName,
                                   const std::string &graphName);
  std::string generateGraphAttr(Attribute graphAttr);
  std::string generateNodeAttr(Operation *op, const std::string &nodeName, int nodeCounter);
  std::string generateTensorAttr(Operation *op);
  std::string generateApiAttr(Operation *op);
  std::string generateIrAttrDef(Operation *op);
  std::string generateIrDef(Operation *op);
  std::string generateInputSrc(Operation *op, const llvm::StringMap<std::string> &ssaToNodeName);
  std::string generateDataNode(int nodeIndex, int argIndex, Type argType, const std::string &graphName,
                               Attribute argAttr = nullptr);
  std::string generateOutputNode(int index, const std::string &lastNodeName, const std::string &graphName);
  int parseDataType(Type type);
  std::string calculateStrides(ArrayRef<int64_t> shape);
};

AFIRToASCIRTextPass::OpInfo AFIRToASCIRTextPass::getOpInfo(StringRef opName) {
  static const llvm::StringMap<OpInfo> opInfoMap = {
      {"load", {1, 2, 11, false}},        {"store", {1, 2, 1, false}},      {"scalar", {1, 1, 11, false}},
      {"index_expr", {1, 1, 11, false}},  {"broadcast", {2, 7, 11, true}},  {"cast", {1, 5, 3, false}},
      {"select", {1, 5, 3, false}},       {"where", {1, 5, 3, false}},      {"concat", {1, 5, 7, false}},
      {"max", {1, 5, 3, false}},          {"min", {1, 5, 3, false}},        {"sum", {1, 5, 3, false}},
      {"mean", {1, 5, 3, false}},         {"prod", {1, 5, 3, false}},       {"any", {1, 5, 3, false}},
      {"all", {1, 5, 3, false}},          {"abs", {1, 5, 3, false}},        {"exp", {1, 5, 3, false}},
      {"ln", {1, 5, 3, false}},           {"sqrt", {1, 5, 3, false}},       {"rsqrt", {1, 5, 3, false}},
      {"reciprocal", {1, 5, 3, false}},   {"erf", {1, 5, 3, false}},        {"tanh", {1, 5, 3, false}},
      {"relu", {1, 5, 3, false}},         {"neg", {1, 5, 3, false}},        {"sigmoid", {1, 5, 3, false}},
      {"logical_not", {1, 5, 3, false}},  {"isnan", {1, 5, 3, false}},      {"isfinite", {1, 5, 3, false}},
      {"leaky_relu", {1, 5, 3, false}},   {"add", {1, 5, 3, false}},        {"sub", {1, 5, 3, false}},
      {"mul", {1, 5, 3, false}},          {"div", {1, 5, 3, false}},        {"minimum", {1, 5, 3, false}},
      {"maximum", {1, 5, 3, false}},      {"truediv", {1, 5, 3, false}},    {"pow", {1, 5, 3, false}},
      {"bitwise_and", {1, 5, 3, false}},  {"floor_div", {1, 5, 3, false}},  {"gelu", {1, 5, 3, false}},
      {"sign", {1, 5, 3, false}},         {"logical_or", {1, 5, 3, false}}, {"logical_and", {1, 5, 3, false}},
      {"ge", {1, 5, 3, false}},           {"eq", {1, 5, 3, false}},         {"ne", {1, 5, 3, false}},
      {"gt", {1, 5, 3, false}},           {"le", {1, 5, 3, false}},         {"lt", {1, 5, 3, false}},
      {"clip_by_value", {1, 5, 3, false}}};

  auto it = opInfoMap.find(opName);
  if (it != opInfoMap.end()) {
    return it->second;
  }

  return {1, 5, 11, false};
}

std::string AFIRToASCIRTextPass::calculateStrides(ArrayRef<int64_t> shape) {
  std::string result;
  llvm::raw_string_ostream os(result);

  int rank = shape.size();
  for (int i = 0; i < rank; i++) {
    if (shape[i] == 1) {
      os << "      strides: \"0\"\n";
    } else {
      int64_t stride = 1;
      for (int j = i + 1; j < rank; j++) {
        if (shape[j] > 0 && shape[j] != 1) {
          stride *= shape[j];
        }
      }
      os << "      strides: \"" << stride << "\"\n";
    }
  }

  os.flush();

  return result;
}

std::string AFIRToASCIRTextPass::convertModuleToASCIRText(ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream os(result);

  Attribute graphAttr = module->getAttr("afir.asc_graph_attr");

  for (Operation &op : module.getBody()->getOperations()) {
    if (auto func = dyn_cast<func::FuncOp>(op)) {
      os << convertFuncToASCIRText(func, graphAttr);
    }
  }

  return result;
}

std::string AFIRToASCIRTextPass::convertFuncToASCIRText(func::FuncOp func, Attribute graphAttr) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string graphName = func.getName().str();

  os << "asc_graph_attr {\n";

  if (graphAttr) {
    os << generateGraphAttr(graphAttr);
  } else {
    os << "  tiling_key: -1\n";
  }

  os << "}\n";

  int nodeCounter = 0;
  llvm::StringMap<std::string> ssaToNodeName;
  llvm::StringMap<int> argToDataCounter;
  llvm::StringMap<Type> argToType;
  int dataCounter = 0;

  // Map arguments to their data counter
  llvm::StringMap<Attribute> argToAttr;

  // Get argument attributes from function
  auto argAttrsAttr = func.getArgAttrsAttr();
  if (argAttrsAttr) {
    for (auto arg : func.getArguments()) {
      std::string ptrStr;
      llvm::raw_string_ostream(ptrStr) << arg.getAsOpaquePointer();
      std::string ssaName = "ssa_" + ptrStr;
      argToDataCounter[ssaName] = dataCounter;
      argToType[ssaName] = arg.getType();

      // Get argument attributes
      if (auto argAttrs = dyn_cast<DictionaryAttr>(argAttrsAttr[dataCounter])) {
        auto outputsAttr = argAttrs.get("outputs");
        if (outputsAttr) {
          auto outputsArrayAttr = dyn_cast<ArrayAttr>(outputsAttr);
          if (outputsArrayAttr && !outputsArrayAttr.empty()) {
            argToAttr[ssaName] = outputsArrayAttr[0];
          }
        }
      }

      dataCounter++;
    }
  } else {
    for (auto arg : func.getArguments()) {
      std::string ptrStr;
      llvm::raw_string_ostream(ptrStr) << arg.getAsOpaquePointer();
      std::string ssaName = "ssa_" + ptrStr;
      argToDataCounter[ssaName] = dataCounter;
      argToType[ssaName] = arg.getType();
      dataCounter++;
    }
  }

  // Generate compute nodes
  Block &block = func.getBody().front();
  for (Operation &op : block) {
    if (!isa<func::ReturnOp>(op)) {
      // Generate Data nodes for arguments that are used for the first time
      for (Value operand : op.getOperands()) {
        std::string ptrStr;
        llvm::raw_string_ostream(ptrStr) << operand.getAsOpaquePointer();
        std::string ssaName = "ssa_" + ptrStr;

        if (argToDataCounter.count(ssaName) && !ssaToNodeName.count(ssaName)) {
          int idx = argToDataCounter[ssaName];
          Type argType = argToType[ssaName];
          Attribute argAttr = argToAttr.count(ssaName) ? argToAttr[ssaName] : nullptr;
          std::string dataNodeName = graphName + "/Data_" + std::to_string(nodeCounter);
          ssaToNodeName[ssaName] = dataNodeName;

          std::string dataNodeText = generateDataNode(nodeCounter, idx, argType, graphName, argAttr);
          os << dataNodeText;
          nodeCounter++;
        }
      }

      std::string nodeText = convertOpToASCIRText(&op, nodeCounter, ssaToNodeName, graphName);
      os << nodeText;
    }
  }

  // Generate Output node
  if (block.getTerminator() && isa<func::ReturnOp>(block.getTerminator())) {
    func::ReturnOp returnOp = cast<func::ReturnOp>(block.getTerminator());
    if (returnOp.getNumOperands() > 0) {
      Value lastValue = returnOp.getOperand(0);
      std::string ptrStr;
      llvm::raw_string_ostream(ptrStr) << lastValue.getAsOpaquePointer();
      std::string ssaName = "ssa_" + ptrStr;

      if (ssaToNodeName.count(ssaName)) {
        os << generateOutputNode(nodeCounter, ssaToNodeName.lookup(ssaName), graphName);
        nodeCounter++;
      }
    }
  }

  os << "graph_name: \"" << graphName << "\"";

  os.flush();

  return result;
}

std::string AFIRToASCIRTextPass::convertOpToASCIRText(Operation *op, int &nodeCounter,
                                                      llvm::StringMap<std::string> &ssaToNodeName,
                                                      const std::string &graphName) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string opName = op->getName().getStringRef().str();
  std::string baseOpName;

  if (opName.find("afir.") == 0) {
    baseOpName = opName.substr(5);
  } else {
    baseOpName = opName;
  }

  std::string nodeName;
  std::string ssaName;

  if (op->getNumResults() > 0) {
    Value result = op->getResult(0);
    std::string ptrStr;
    llvm::raw_string_ostream(ptrStr) << result.getAsOpaquePointer();
    ssaName = "ssa_" + ptrStr;
  }

  std::string capitalizedOpName = baseOpName;
  if (!baseOpName.empty()) {
    capitalizedOpName[0] = ::toupper(baseOpName[0]);
  }

  if (baseOpName == "load") {
    nodeName = "Load_" + std::to_string(nodeCounter);
  } else if (baseOpName == "store") {
    nodeName = "Store_" + std::to_string(nodeCounter);
  } else {
    nodeName = capitalizedOpName + "_" + std::to_string(nodeCounter);
  }

  std::string fullNodeName = graphName + "/" + nodeName;

  if (!ssaName.empty()) {
    ssaToNodeName[ssaName] = fullNodeName;
  }

  os << "asc_node {\n";
  os << generateInputSrc(op, ssaToNodeName);
  os << generateTensorAttr(op);
  os << generateNodeAttr(op, fullNodeName, nodeCounter);
  os << generateIrDef(op);
  os << "}\n";

  nodeCounter++;

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateGraphAttr(Attribute graphAttr) {
  std::string result;
  llvm::raw_string_ostream os(result);

  auto ascGraphAttr = dyn_cast<afir::AscGraphAttrGroupsAttr>(graphAttr);
  if (!ascGraphAttr) {
    return result;
  }

  int64_t tilingKey = ascGraphAttr.getTilingKey();
  os << "  tiling_key: " << tilingKey << "\n";

  auto axes = ascGraphAttr.getAxes();

  for (auto axis : axes) {
    int64_t id = axis.getId();
    std::string name = axis.getName().str();
    std::string size = axis.getSize().str();
    std::string align = axis.getAlign() ? axis.getAlign().str() : "1";

    os << "  axis {\n";
    if (id > 0) {
      os << "    id: " << id << "\n";
    }
    os << "    name: \"" << name << "\"\n";
    os << "    size: \"" << size << "\"\n";
    os << "    align: \"" << align << "\"\n";
    os << "    allow_unaligned_tail: true\n";
    os << "  }\n";
  }

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateNodeAttr(Operation *op, const std::string &nodeName, int nodeCounter) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string opName = op->getName().getStringRef().str();
  std::string baseOpName = opName.find("afir.") == 0 ? opName.substr(5) : opName;

  std::string capitalizedOpName = baseOpName;
  if (!baseOpName.empty()) {
    capitalizedOpName[0] = ::toupper(baseOpName[0]);
  }

  OpInfo opInfo = getOpInfo(baseOpName);

  int execOrder = nodeCounter;
  if (opInfo.isBroadcast) {
    execOrder = -1;
  } else if (baseOpName == "load") {
    execOrder = nodeCounter;
  } else {
    execOrder = nodeCounter - 1;
  }

  os << "  attr {\n";
  os << "    name: \"" << nodeName << "\"\n";
  os << "    type: \"" << capitalizedOpName << "\"\n";
  os << "    sched {\n";
  os << "      exec_order: " << execOrder << "\n";

  auto indexingMapsAttr = op->getAttrOfType<ArrayAttr>("indexing_maps");
  if (indexingMapsAttr) {
    int numOperands = op->getNumOperands();

    if (numOperands < static_cast<int>(indexingMapsAttr.size())) {
      if (auto affineMapAttr = dyn_cast<AffineMapAttr>(indexingMapsAttr[numOperands])) {
        auto affineMap = affineMapAttr.getValue();
        for (unsigned i = 0; i < affineMap.getNumResults(); i++) {
          auto expr = affineMap.getResult(i);
          if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
            os << "      axis: " << dimExpr.getPosition() << "\n";
          } else {
            os << "      axis: " << i << "\n";
          }
        }
      }
    }
  }

  auto loopAxisAttr = op->getAttrOfType<IntegerAttr>("loop_axis");
  int32_t loopAxis = loopAxisAttr ? loopAxisAttr.getInt() : -1;
  os << "      loop_axis: " << loopAxis << "\n";

  os << "    }\n";
  os << generateApiAttr(op);
  os << generateIrAttrDef(op);
  os << "  }\n";

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateTensorAttr(Operation *op) {
  std::string result;
  llvm::raw_string_ostream os(result);

  auto outputsAttr = op->getAttrOfType<ArrayAttr>("outputs");
  if (!outputsAttr || outputsAttr.empty()) {
    os.flush();
    return result;
  }

  auto ascTensorAttr = dyn_cast<afir::AscTensorGroupsAttr>(outputsAttr[0]);
  if (!ascTensorAttr) {
    os.flush();
    return result;
  }

  int64_t tensorId = ascTensorAttr.getTensorId();
  int64_t reuseId = ascTensorAttr.getReuseId();
  int64_t positionId = ascTensorAttr.getPositionId();

  auto positionConfig = ascTensorAttr.getPositionConfig();
  int64_t queDepth = -1;
  int queBufNum = -1;

  uint32_t depth = positionConfig.getDepth();
  bool isDoubleBuffer = positionConfig.getIsDoubleBuffer();

  if (depth > 0) {
    queDepth = depth;
  }
  if (isDoubleBuffer) {
    queBufNum = 2;
  }

  Type tensorType = op->getResult(0).getType();
  if (auto shapedType = dyn_cast<ShapedType>(tensorType)) {
    int rank = shapedType.getRank();
    auto shape = shapedType.getShape();

    os << "  outputs {\n";
    os << "    attr {\n";

    for (int i = 0; i < rank; i++) {
      os << "      axis_ids: " << i << "\n";
    }

    for (int i = 0; i < rank; i++) {
      int64_t size = shapedType.getDimSize(i);
      if (size == ShapedType::kDynamic) {
        os << "      repeats: \"-1\"\n";
      } else {
        os << "      repeats: \"" << size << "\"\n";
      }
    }

    os << calculateStrides(shape);

    os << "      mem {\n";
    os << "        tensor_id: " << tensorId << "\n";
    os << "      }\n";
    os << "      que {\n";
    os << "        id: -1\n";
    os << "        depth: " << queDepth << "\n";
    os << "        buf_num: " << queBufNum << "\n";
    os << "      }\n";
    os << "      buf {\n";
    os << "        id: " << positionId << "\n";
    os << "      }\n";
    os << "      opt {\n";
    os << "        reuse_id: " << reuseId << "\n";
    os << "        ref_tensor: -1\n";
    os << "        merge_scope: -1\n";
    os << "      }\n";
    os << "    }\n";
    os << "  }\n";
  }

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateApiAttr(Operation *op) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string opName = op->getName().getStringRef().str();
  std::string baseOpName;

  if (opName.find("afir.") == 0) {
    baseOpName = opName.substr(5);
  } else {
    baseOpName = opName;
  }

  OpInfo opInfo = getOpInfo(baseOpName);

  os << "    api {\n";

  auto apiTypeAttr = op->getAttrOfType<IntegerAttr>("api_type");
  int apiType = apiTypeAttr ? apiTypeAttr.getInt() : opInfo.apiType;
  os << "      type: " << apiType << "\n";

  auto computeTypeAttr = op->getAttrOfType<IntegerAttr>("compute_type");
  int computeType = computeTypeAttr ? computeTypeAttr.getInt() : opInfo.computeType;
  if (computeType > 0 && baseOpName != "load") {
    os << "      compute_type: " << computeType << "\n";
  }

  auto apiUnitAttr = op->getAttrOfType<IntegerAttr>("api_unit");
  int apiUnit = apiUnitAttr ? apiUnitAttr.getInt() : opInfo.apiUnit;
  os << "      unit: " << apiUnit << "\n";

  os << "    }\n";

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateIrAttrDef(Operation *op) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string opName = op->getName().getStringRef().str();
  std::string baseOpName;

  if (opName.find("afir.") == 0) {
    baseOpName = opName.substr(5);
  } else {
    baseOpName = opName;
  }

  bool needsEmptyIrAttrDef = (baseOpName == "load" || baseOpName == "store" || baseOpName == "scalar");
  auto irAttrDefAttr = op->getAttrOfType<DictionaryAttr>("ir_attr_def");

  if (needsEmptyIrAttrDef || (irAttrDefAttr && !irAttrDefAttr.empty())) {
    os << "    ir_attr_def {\n";

    if (baseOpName == "scalar") {
      auto valueAttr = op->getAttr("value");
      if (valueAttr) {
        os << "      attr {\n";
        os << "        key: \"value\"\n";
        os << "        value {\n";

        if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
          os << "          i: " << intAttr.getInt() << "\n";
        } else if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
          std::ostringstream oss;
          oss << std::scientific << std::setprecision(17) << floatAttr.getValueAsDouble();
          os << "          s: \"" << oss.str() << "\"\n";
        }

        os << "        }\n";
        os << "      }\n";
      }
    }

    if (irAttrDefAttr && !irAttrDefAttr.empty()) {
      for (auto attr : irAttrDefAttr) {
        os << "      attr {\n";
        os << "        key: \"" << attr.getName().strref() << "\"\n";
        os << "        value {\n";

        auto value = attr.getValue();
        if (auto stringAttr = dyn_cast<StringAttr>(value)) {
          os << "          expression: \"" << stringAttr.getValue() << "\"\n";
        } else if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
          os << "          i: " << intAttr.getInt() << "\n";
        } else if (auto boolAttr = dyn_cast<BoolAttr>(value)) {
          os << "          b: " << (boolAttr.getValue() ? "true" : "false") << "\n";
        } else if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
          os << "          f: " << floatAttr.getValueAsDouble() << "\n";
        } else {
          os << "          s: \"" << value << "\"\n";
        }

        os << "        }\n";
        os << "      }\n";
      }
    }

    os << "    }\n";
  }

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateIrDef(Operation *op) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string opName = op->getName().getStringRef().str();
  std::string baseOpName;

  if (opName.find("afir.") == 0) {
    baseOpName = opName.substr(5);
  } else {
    baseOpName = opName;
  }

  int inputNums = op->getNumOperands();
  int outputNums = op->getNumResults();

  os << "  ir_def {\n";

  for (int i = 0; i < inputNums; i++) {
    if (inputNums == 1) {
      os << "    input_names: \"x\"\n";
    } else {
      os << "    input_names: \"x" << (i + 1) << "\"\n";
    }
  }

  for (int i = 0; i < outputNums; i++) {
    os << "    output_names: \"y\"\n";
  }

  for (int i = 0; i < inputNums; i++) {
    os << "    input_ir_type: 0\n";
  }

  for (int i = 0; i < outputNums; i++) {
    os << "    output_ir_type: 0\n";
  }

  std::string capitalizedOpName = baseOpName;
  if (!baseOpName.empty()) {
    capitalizedOpName[0] = ::toupper(baseOpName[0]);
  }

  os << "    type: \"" << capitalizedOpName << "\"\n";

  for (int i = 0; i < inputNums; i++) {
    os << "    input_nums: 1\n";
  }

  for (int i = 0; i < outputNums; i++) {
    os << "    output_nums: 1\n";
  }

  os << "  }\n";

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateInputSrc(Operation *op, const llvm::StringMap<std::string> &ssaToNodeName) {
  std::string result;
  llvm::raw_string_ostream os(result);

  for (Value operand : op->getOperands()) {
    std::string ptrStr;
    llvm::raw_string_ostream(ptrStr) << operand.getAsOpaquePointer();
    std::string ssaName = "ssa_" + ptrStr;
    if (ssaToNodeName.count(ssaName)) {
      os << "  input_src {\n";
      os << "    src_node_name: \"" << ssaToNodeName.lookup(ssaName) << "\"\n";
      os << "  }\n";
    }
  }

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateDataNode(int nodeIndex, int argIndex, Type argType,
                                                  const std::string &graphName, Attribute argAttr) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string nodeName = graphName + "/Data_" + std::to_string(nodeIndex);

  os << "asc_node {\n";
  os << "  outputs {\n";
  os << "    attr {\n";

  int64_t tensorId = -1;
  int64_t reuseId = -1;
  int64_t positionId = -1;
  int64_t queDepth = -1;
  int queBufNum = -1;

  if (argAttr) {
    auto ascTensorAttr = dyn_cast<afir::AscTensorGroupsAttr>(argAttr);
    if (ascTensorAttr) {
      tensorId = ascTensorAttr.getTensorId();
      reuseId = ascTensorAttr.getReuseId();
      positionId = ascTensorAttr.getPositionId();

      auto positionConfig = ascTensorAttr.getPositionConfig();
      uint32_t depth = positionConfig.getDepth();
      bool isDoubleBuffer = positionConfig.getIsDoubleBuffer();

      if (depth > 0) {
        queDepth = depth;
      }
      if (isDoubleBuffer) {
        queBufNum = 2;
      }
    }
  }

  if (auto shapedType = dyn_cast<ShapedType>(argType)) {
    int rank = shapedType.getRank();
    auto shape = shapedType.getShape();

    for (int i = 0; i < rank; i++) {
      os << "      axis_ids: " << i << "\n";
    }

    for (int i = 0; i < rank; i++) {
      int64_t size = shapedType.getDimSize(i);
      if (size == ShapedType::kDynamic) {
        os << "      repeats: \"-1\"\n";
      } else {
        os << "      repeats: \"" << size << "\"\n";
      }
    }

    os << calculateStrides(shape);
  } else {
    os << "      axis_ids: 0\n";
    os << "      repeats: \"1\"\n";
    os << "      strides: \"1\"\n";
  }

  os << "      mem {\n";
  os << "        tensor_id: " << tensorId << "\n";
  os << "      }\n";
  os << "      que {\n";
  os << "        id: -1\n";
  os << "        depth: " << queDepth << "\n";
  os << "        buf_num: " << queBufNum << "\n";
  os << "      }\n";
  os << "      buf {\n";
  os << "        id: " << positionId << "\n";
  os << "      }\n";
  os << "      opt {\n";
  os << "        reuse_id: " << reuseId << "\n";
  os << "        ref_tensor: -1\n";
  os << "        merge_scope: -1\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "  attr {\n";
  os << "    name: \"" << nodeName << "\"\n";
  os << "    type: \"Data\"\n";
  os << "    sched {\n";
  if (nodeIndex > 0) {
    os << "      exec_order: " << nodeIndex << "\n";
  }

  if (auto shapedType = dyn_cast<ShapedType>(argType)) {
    int rank = shapedType.getRank();
    for (int i = 0; i < rank; i++) {
      os << "      axis: " << i << "\n";
    }
  } else {
    os << "      axis: 0\n";
  }

  os << "      loop_axis: -1\n";
  os << "    }\n";
  os << "    api {\n";
  os << "      compute_type: 11\n";
  os << "    }\n";
  os << "    ir_attr_def {\n";
  os << "      attr {\n";
  os << "        key: \"index\"\n";
  os << "        value {\n";
  os << "          i: " << argIndex << "\n";
  os << "        }\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "  ir_def {\n";
  os << "    output_names: \"y\"\n";
  os << "    output_ir_type: 0\n";
  os << "    type: \"Data\"\n";
  os << "    output_nums: 1\n";
  os << "  }\n";
  os << "}\n";

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateOutputNode(int index, const std::string &lastNodeName,
                                                    const std::string &graphName) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string nodeName = graphName + "/Output_" + std::to_string(index);

  os << "asc_node {\n";
  os << "  input_src {\n";
  os << "    src_node_name: \"" << lastNodeName << "\"\n";
  os << "  }\n";
  os << "  outputs {\n";
  os << "    attr {\n";
  os << "      mem {\n";
  os << "        tensor_id: -1\n";
  os << "      }\n";
  os << "      que {\n";
  os << "        id: -1\n";
  os << "        depth: -1\n";
  os << "        buf_num: -1\n";
  os << "      }\n";
  os << "      buf {\n";
  os << "        id: -1\n";
  os << "      }\n";
  os << "      opt {\n";
  os << "        reuse_id: -1\n";
  os << "        ref_tensor: -1\n";
  os << "        merge_scope: -1\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "  attr {\n";
  os << "    name: \"" << nodeName << "\"\n";
  os << "    type: \"Output\"\n";
  os << "    sched {\n";
  os << "      exec_order: 8\n";
  os << "      axis: 0\n";
  os << "      axis: 1\n";
  os << "      loop_axis: -1\n";
  os << "    }\n";
  os << "    api {\n";
  os << "      compute_type: 11\n";
  os << "    }\n";
  os << "    ir_attr_def {\n";
  os << "      attr {\n";
  os << "        key: \"index\"\n";
  os << "        value {\n";
  os << "          i: 0\n";
  os << "        }\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "  ir_def {\n";
  os << "    input_names: \"x\"\n";
  os << "    output_names: \"y\"\n";
  os << "    input_ir_type: 0\n";
  os << "    output_ir_type: 0\n";
  os << "    type: \"Output\"\n";
  os << "    input_nums: 1\n";
  os << "    output_nums: 1\n";
  os << "  }\n";
  os << "}\n";

  os.flush();
  return result;
}

int AFIRToASCIRTextPass::parseDataType(Type type) {
  return llvm::TypeSwitch<Type, int>(type)
      .Case<Float32Type>([](Float32Type) { return 1; })
      .Case<Float16Type>([](Float16Type) { return 2; })
      .Case<Float64Type>([](Float64Type) { return 12; })
      .Case<IntegerType>([](IntegerType intType) {
        if (intType.getWidth() == 8) {
          return intType.isUnsigned() ? 4 : 3;
        }
        if (intType.getWidth() == 16) {
          return intType.isUnsigned() ? 6 : 5;
        }
        if (intType.getWidth() == 32) {
          return intType.isUnsigned() ? 9 : 7;
        }
        if (intType.getWidth() == 64) {
          return intType.isUnsigned() ? 10 : 8;
        }
        return 1;
      })
      .Default([](Type) { return 1; });
}

std::unique_ptr<Pass> createConvertAFIRToASCIRTextPass() {
  return std::unique_ptr<Pass>(new AFIRToASCIRTextPass());
}

}  // namespace afir
}  // namespace mlir
