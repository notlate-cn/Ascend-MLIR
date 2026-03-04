//===- AFIRToASCIRText.cpp - AFIR to ASCIR text conversion ----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AFIRToASCIR/AFIRToASCIRText.h"
#include "Dialect/AFIR/AFIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

using namespace mlir;
using namespace afir;

namespace mlir::afir {

#define GEN_PASS_DECL_CONVERTAFIRTOASCIRTEXTPASS
#define GEN_PASS_DEF_CONVERTAFIRTOASCIRTEXTPASS
#include "Conversion/Passes.h.inc"

namespace {

struct AFIRToASCIRTextPass : public impl::ConvertAFIRToASCIRTextPassBase<AFIRToASCIRTextPass> {
  using impl::ConvertAFIRToASCIRTextPassBase<AFIRToASCIRTextPass>::ConvertAFIRToASCIRTextPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string output = convertModuleToASCIRText(module);

    if (!ascirPath.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream os(ascirPath, ec);
      if (ec) {
        llvm::errs() << "Error opening file " << ascirPath << ": " << ec.message() << "\n";
        signalPassFailure();
        return;
      }
      os << output;
      os.close();
    } else {
      llvm::errs() << output;
    }
  }

 private:
  std::string convertModuleToASCIRText(ModuleOp module) {
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

  std::string convertFuncToASCIRText(func::FuncOp func, Attribute graphAttr) {
    std::string result;
    llvm::raw_string_ostream os(result);

    std::string graphName = func.getName().str();

    os << "asc_graph_attr {\n";

    if (graphAttr) {
      os << generateGraphAttr(graphAttr);
    } else {
      os << "  type: Compute\n";
    }

    os << "}\n";

    int nodeCounter = 0;
    int execOrderCounter = 0;
    llvm::StringMap<std::string> ssaToNodeName;

    // Track which arguments have been converted to Data nodes
    llvm::DenseSet<unsigned> convertedArgs;

    // Generate compute nodes and Data nodes on-demand
    Block &block = func.getBody().front();
    std::string lastNodeName;
    for (Operation &op : block) {
      if (!isa<func::ReturnOp>(op)) {
        // Generate Data nodes for operands that haven't been converted yet
        for (Value operand : op.getOperands()) {
          if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
            unsigned argIndex = blockArg.getArgNumber();
            if (!convertedArgs.contains(argIndex)) {
              std::string dataNodeName = graphName + "/Data_" + std::to_string(nodeCounter);
              std::string ptrStr;
              llvm::raw_string_ostream(ptrStr) << operand.getAsOpaquePointer();
              std::string ssaName = "ssa_" + ptrStr;
              ssaToNodeName[ssaName] = dataNodeName;

              Type argType = operand.getType();
              std::string dataNodeText = generateDataNode(nodeCounter, argIndex, argType, graphName, execOrderCounter);
              os << dataNodeText;
              nodeCounter++;
              execOrderCounter++;
              convertedArgs.insert(argIndex);
            }
          }
        }

        std::string nodeText = convertOpToASCIRText(&op, nodeCounter, execOrderCounter, ssaToNodeName, graphName);
        os << nodeText;
        if (op.getNumResults() > 0) {
          std::string ptrStr;
          llvm::raw_string_ostream(ptrStr) << op.getResult(0).getAsOpaquePointer();
          std::string ssaName = "ssa_" + ptrStr;
          if (ssaToNodeName.count(ssaName)) {
            lastNodeName = ssaToNodeName.lookup(ssaName);
          }
        }
      }
    }

    // Generate Output node
    if (!lastNodeName.empty()) {
      std::string outputNodeText = generateOutputNode(nodeCounter, execOrderCounter, lastNodeName, graphName);
      os << outputNodeText;
    }

    os << "graph_name: \"" << graphName << "\"";

    return result;
  }

  std::string convertOpToASCIRText(Operation *op, int &nodeCounter, int &execOrderCounter,
                                   llvm::StringMap<std::string> &ssaToNodeName, const std::string &graphName) {
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

    if (baseOpName == "load") {
      nodeName = "Load_" + std::to_string(nodeCounter);
    } else if (baseOpName == "store") {
      nodeName = "Store_" + std::to_string(nodeCounter);
    } else if (baseOpName == "broadcast") {
      nodeName = "Broadcast_" + std::to_string(nodeCounter);
    } else {
      std::string capitalizedOpName = baseOpName;
      if (!baseOpName.empty()) {
        capitalizedOpName[0] = ::toupper(baseOpName[0]);
      }
      nodeName = capitalizedOpName + "_" + std::to_string(nodeCounter);
    }

    std::string fullNodeName = graphName + "/" + nodeName;

    if (!ssaName.empty()) {
      ssaToNodeName[ssaName] = fullNodeName;
    }

    os << "asc_node {\n";
    os << generateInputSrc(op, ssaToNodeName);
    os << generateTensorAttr(op->getResult(0).getType());
    os << generateNodeAttr(op, fullNodeName, execOrderCounter);
    os << generateIrDef(op);
    os << "}\n";

    nodeCounter++;

    return result;
  }

  std::string generateGraphAttr(Attribute graphAttr) {
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

    return result;
  }

  std::string generateNodeAttr(Operation *op, const std::string &nodeName, int &execOrderCounter) {
    std::string result;
    llvm::raw_string_ostream os(result);

    std::string opName = op->getName().getStringRef().str();
    std::string baseOpName;

    if (opName.find("afir.") == 0) {
      baseOpName = opName.substr(5);
    } else {
      baseOpName = opName;
    }

    int execOrder = execOrderCounter;
    if (baseOpName == "broadcast") {
      execOrder = -1;
    } else {
      execOrderCounter++;
    }

    std::string capitalizedOpName = baseOpName;
    if (!baseOpName.empty()) {
      capitalizedOpName[0] = ::toupper(baseOpName[0]);
    }

    os << "  attr {\n";
    os << "    name: \"" << nodeName << "\"\n";
    os << "    type: \"" << capitalizedOpName << "\"\n";
    os << "    sched {\n";
    os << "      exec_order: " << execOrder << "\n";
    os << "      axis: 0\n";
    os << "      axis: 1\n";
    os << "      loop_axis: -1\n";
    os << "    }\n";
    os << generateApiAttr(op);
    os << generateIrAttrDef(op);
    os << "  }\n";

    return result;
  }

  std::string generateTensorAttr(Type tensorType) {
    std::string result;
    llvm::raw_string_ostream os(result);

    if (auto shapedType = dyn_cast<ShapedType>(tensorType)) {
      os << "  outputs {\n";
      os << "    attr {\n";

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
    }

    return result;
  }

  std::string generateSchedAttr(int execOrder, ArrayRef<int> axisIds) {
    std::string result;
    llvm::raw_string_ostream os(result);

    os << "    sched {\n";
    os << "      exec_order: " << execOrder << "\n";
    for (int axisId : axisIds) {
      os << "      axis: " << axisId << "\n";
    }
    os << "      loop_axis: -1\n";
    os << "    }\n";

    return result;
  }

  std::string generateApiAttr(Operation *op) {
    std::string result;
    llvm::raw_string_ostream os(result);

    std::string opName = op->getName().getStringRef().str();
    std::string baseOpName;

    if (opName.find("afir.") == 0) {
      baseOpName = opName.substr(5);
    } else {
      baseOpName = opName;
    }

    int apiType = 1;
    int apiUnit = 7;
    int computeType = 0;

    if (baseOpName == "load") {
      apiType = 1;
      apiUnit = 2;
    } else if (baseOpName == "store") {
      apiType = 1;
      computeType = 1;
      apiUnit = 2;
    } else if (baseOpName == "broadcast") {
      apiType = 2;
      computeType = 11;
    } else if (baseOpName == "add") {
      apiType = 1;
      computeType = 3;
      apiUnit = 5;
    } else if (baseOpName == "sub" || baseOpName == "mul") {
      apiType = 1;
      computeType = 3;
      apiUnit = 5;
    }

    os << "    api {\n";
    os << "      type: " << apiType << "\n";
    if (computeType > 0) {
      os << "      compute_type: " << computeType << "\n";
    }
    os << "      unit: " << apiUnit << "\n";
    os << "    }\n";

    return result;
  }

  std::string generateIrAttrDef(Operation *op) {
    std::string result;
    llvm::raw_string_ostream os(result);

    std::string opName = op->getName().getStringRef().str();
    std::string baseOpName;

    if (opName.find("afir.") == 0) {
      baseOpName = opName.substr(5);
    } else {
      baseOpName = opName;
    }

    if (baseOpName == "load") {
      os << "    ir_attr_def {\n";
      os << "      attr {\n";
      os << "        key: \"offset\"\n";
      os << "        value {\n";
      os << "          expression: \"0\"\n";
      os << "        }\n";
      os << "      }\n";
      os << "    }\n";
    } else if (baseOpName == "store") {
      os << "    ir_attr_def {\n";
      os << "    }\n";
    }

    return result;
  }

  std::string generateIrDef(Operation *op) {
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
    int inputIrType = 0;
    int outputIrType = 0;

    std::string capitalizedOpName = baseOpName;
    if (!baseOpName.empty()) {
      capitalizedOpName[0] = ::toupper(baseOpName[0]);
    }

    os << "  ir_def {\n";
    if (baseOpName == "load" || baseOpName == "store" || baseOpName == "broadcast" || baseOpName == "Output") {
      os << "    input_names: \"x\"\n";
    } else {
      for (int i = 0; i < inputNums; i++) {
        os << "    input_names: \"x" << (i + 1) << "\"\n";
      }
    }
    os << "    output_names: \"y\"\n";
    if (baseOpName == "load" || baseOpName == "store" || baseOpName == "broadcast" || baseOpName == "Output") {
      os << "    input_ir_type: " << inputIrType << "\n";
    } else {
      for (int i = 0; i < inputNums; i++) {
        os << "    input_ir_type: " << inputIrType << "\n";
      }
    }
    os << "    output_ir_type: " << outputIrType << "\n";
    os << "    type: \"" << capitalizedOpName << "\"\n";
    if (baseOpName == "load" || baseOpName == "store" || baseOpName == "broadcast" || baseOpName == "Output") {
      os << "    input_nums: " << inputNums << "\n";
    } else {
      for (int i = 0; i < inputNums; i++) {
        os << "    input_nums: 1\n";
      }
    }
    os << "    output_nums: " << outputNums << "\n";
    os << "  }\n";

    return result;
  }

  std::string generateInputSrc(Operation *op, const llvm::StringMap<std::string> &ssaToNodeName) {
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

    return result;
  }

  std::string calculateStrides(ArrayRef<int64_t> shape) {
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

  std::string generateDataNode(int index, int argIndex, Type argType, const std::string &graphName, int execOrder) {
    std::string result;
    llvm::raw_string_ostream os(result);

    std::string nodeName = graphName + "/Data_" + std::to_string(index);

    os << "asc_node {\n";
    os << "  outputs {\n";
    os << "    attr {\n";

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
    os << "    type: \"Data\"\n";
    os << "    sched {\n";
    if (execOrder > 0) {
      os << "      exec_order: " << execOrder << "\n";
    }
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

  std::string generateOutputNode(int index, int execOrder, const std::string &lastNodeName, const std::string &graphName) {
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
    os << "      exec_order: " << execOrder << "\n";
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
};

}  // namespace

std::unique_ptr<Pass> createConvertAFIRToASCIRTextPass() {
  return std::make_unique<AFIRToASCIRTextPass>();
}

}  // namespace mlir::afir
