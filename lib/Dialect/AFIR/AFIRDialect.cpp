//===- AFIRDialect.cpp - AFIR dialect implementation ------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIRDialect.h"
#include <cstdint>
#include "Dialect/AFIR/AFIROps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::afir;

//===----------------------------------------------------------------------===//
// AFIR dialect initialization
//===----------------------------------------------------------------------===//

// Include generated dialect implementation
#include "Dialect/AFIR/AFIRDialect.cpp.inc"

// Include generated enum definitions
#include "Dialect/AFIR/AFIREnums.cpp.inc"

// Include generated attribute definitions
#define GET_ATTRDEF_CLASSES
#include "Dialect/AFIR/AFIRAttrs.cpp.inc"

// Include generated type definitions
#define GET_TYPEDEF_CLASSES
#include "Dialect/AFIR/AFIRTypes.cpp.inc"

void AFIRDialect::initialize() {
  // Register types
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/AFIR/AFIRTypes.cpp.inc"
      >();

  // Register attributes
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/AFIR/AFIRAttrs.cpp.inc"
      >();

  // Register operations
  addOperations<
#define GET_OP_LIST
#include "Dialect/AFIR/AFIROps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type Parsing and Printing
//===----------------------------------------------------------------------===//

Type AFIRDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeTag;
  if (failed(parser.parseKeyword(&typeTag)))
    return Type();

  if (typeTag != "tensor") {
    parser.emitError(parser.getCurrentLocation(), "unknown afir type: ") << typeTag;
    return Type();
  }

  if (parser.parseLess())
    return Type();

  Type elementType;
  if (parser.parseType(elementType))
    return Type();

  if (parser.parseComma())
    return Type();

  if (parser.parseKeyword("tid="))
    return Type();

  int64_t tensorId;
  if (parser.parseInteger(tensorId))
    return Type();

  if (parser.parseComma())
    return Type();

  if (parser.parseLParen())
    return Type();

  SmallVector<int64_t> shape;
  if (parser.parseDimensionList(shape, /*allowDynamic=*/false))
    return Type();

  if (parser.parseRParen())
    return Type();

  if (parser.parseComma())
    return Type();

  Attribute posAttr = PositionInfoAttr::parse(parser, Type());
  if (!posAttr)
    return Type();

  if (parser.parseGreater())
    return Type();

  return AFIRTensorType::get(getContext(), shape, elementType, tensorId, llvm::cast<PositionInfoAttr>(posAttr));
}

void AFIRDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (auto tensorType = llvm::dyn_cast<AFIRTensorType>(type)) {
    printer << "tensor<";
    printer.printType(mlir::RankedTensorType::get(tensorType.getShape(), tensorType.getElementType()));
    printer << ", tid=" << tensorType.getTensorId() << ", ";
    tensorType.getPosition().print(printer);
    printer << ">";
    return;
  }

  llvm_unreachable("unknown AFIR type");
}

//===----------------------------------------------------------------------===//
// Attribute Parsing and Printing
//===----------------------------------------------------------------------===//

Attribute AFIRDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  StringRef attrTag;
  Attribute attr;
  
  auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
  if (parseResult.has_value()) {
    return attr;
  }
  
  if (attrTag == "pos") {
    return PositionInfoAttr::parse(parser, type);
  }

  parser.emitError(parser.getCurrentLocation(), "unknown afir attribute: ") << attrTag;
  return Attribute();
}

void AFIRDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  if (auto posAttr = llvm::dyn_cast<PositionInfoAttr>(attr)) {
    posAttr.print(printer);
    return;
  }
  
  LogicalResult result = generatedAttributePrinter(attr, printer);
  (void)result;
  assert(succeeded(result));
}

//===----------------------------------------------------------------------===//
// PositionInfoAttr Implementation
//===----------------------------------------------------------------------===//

Attribute PositionInfoAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return Attribute();

  int64_t rid;
  if (parser.parseKeyword("rid=") || parser.parseInteger(rid))
    return Attribute();

  if (parser.parseComma())
    return Attribute();

  StringRef positionStr;
  if (parser.parseKeyword(&positionStr))
    return Attribute();

  Position position;
  if (positionStr == "VEC_IN") {
    position = Position::VEC_IN;
  } else if (positionStr == "VEC_OUT") {
    position = Position::VEC_OUT;
  } else if (positionStr == "VEC_CALC") {
    position = Position::VEC_CALC;
  } else if (positionStr == "L1") {
    position = Position::L1;
  } else if (positionStr == "L0_A") {
    position = Position::L0_A;
  } else if (positionStr == "L0_B") {
    position = Position::L0_B;
  } else if (positionStr == "L0_C") {
    position = Position::L0_C;
  } else if (positionStr == "GM") {
    position = Position::GM;
  } else {
    parser.emitError(parser.getCurrentLocation(), "unknown position type: ") << positionStr;
    return Attribute();
  }

  int64_t depth = 2;
  int64_t buf_num = -1;
  bool db = false;

  if (position == Position::VEC_IN || position == Position::VEC_OUT) {
    if (parser.parseComma())
      return Attribute();

    if (parser.parseKeyword("depth=") || parser.parseInteger(depth))
      return Attribute();

    if (parser.parseComma())
      return Attribute();
    
    if (parser.parseKeyword("buf_num=") || parser.parseInteger(buf_num))
      return Attribute();

    if (parser.parseComma())
      return Attribute();

    StringRef dbStr;
    if (parser.parseKeyword(&dbStr))
      return Attribute();
    if (dbStr != "db") {
      parser.emitError(parser.getCurrentLocation(), "expected 'db' keyword");
      return Attribute();
    }
    db = true;
  }

  if (parser.parseGreater())
    return Attribute();

  return PositionInfoAttr::get(parser.getContext(), rid, position, depth, buf_num, db);
}

void PositionInfoAttr::print(AsmPrinter &printer) const {
  printer << "afir.pos<rid=" << getRid() << ", ";

  Position position = getPosition();
  switch (position) {
    case Position::VEC_IN:
      printer << "VEC_IN";
      break;
    case Position::VEC_OUT:
      printer << "VEC_OUT";
      break;
    case Position::VEC_CALC:
      printer << "VEC_CALC";
      break;
    case Position::L1:
      printer << "L1";
      break;
    case Position::L0_A:
      printer << "L0_A";
      break;
    case Position::L0_B:
      printer << "L0_B";
      break;
    case Position::L0_C:
      printer << "L0_C";
      break;
    case Position::GM:
      printer << "GM";
      break;
  }

  if (position == Position::VEC_IN || position == Position::VEC_OUT) {
    printer << ", depth=" << getDepth();
    printer << ", buf_num=" << getBufNum();
    if (getDb()) {
      printer << ", db";
    }
  }

  printer << ">";
}

AFIRTensorType AFIRTensorType::get(ArrayRef<int64_t> shape, Type elementType,
                                   int64_t tensorId, PositionInfoAttr position) {
  return get(elementType.getContext(), shape, elementType, tensorId, position);
}

//===----------------------------------------------------------------------===//
// AFIRTensorType ShapedTypeInterface Implementation
//===----------------------------------------------------------------------===//

bool AFIRTensorType::hasRank() const {
  return true;
}

ShapedType AFIRTensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                     Type elementType) const {
  if (shape) {
    return AFIRTensorType::get(getContext(), *shape,
                          elementType ? elementType : getElementType(),
                          getTensorId(), getPosition());
  }
  return AFIRTensorType::get(getContext(), getShape(),
                        elementType ? elementType : getElementType(),
                        getTensorId(), getPosition());
}