
#include "mlir/CAPI/Registration.h"
#include "CAPI/Dialects.h"
#include "Dialect/AFIR/AFIR.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AFIR, afir, mlir::afir::AFIRDialect);

bool mlirAttributeIsAFIRPositionConfigAttr(MlirAttribute attr) {
  return llvm::isa<mlir::afir::PositionConfigAttr>(unwrap(attr));
}

MlirAttribute mlirPositionConfigAttrGet(MlirContext mlirCtx, mlir::afir::Position position, uint32_t depth,
                                        bool is_double_buffer) {
  return wrap(mlir::afir::PositionConfigAttr::get(unwrap(mlirCtx), position, depth, is_double_buffer));
}

mlir::afir::Position mlirPositionConfigAttrGetPosition(MlirAttribute attr) {
  mlir::afir::PositionConfigAttr position_config = llvm::cast<mlir::afir::PositionConfigAttr>(unwrap(attr));
  return position_config.getPosition();
}

uint32_t mlirPositionConfigAttrGetDepth(MlirAttribute attr) {
  mlir::afir::PositionConfigAttr position_config = llvm::cast<mlir::afir::PositionConfigAttr>(unwrap(attr));
  return position_config.getDepth();
}

bool mlirPositionConfigAttrGetIsDoubleBuffer(MlirAttribute attr) {
  mlir::afir::PositionConfigAttr position_config = llvm::cast<mlir::afir::PositionConfigAttr>(unwrap(attr));
  return position_config.getIsDoubleBuffer();
}

bool mlirAttributeIsAFIRAscTensorGroupsAttr(MlirAttribute attr) {
  return llvm::isa<mlir::afir::AscTensorGroupsAttr>(unwrap(attr));
}

MlirAttribute mlirAscTensorGroupsAttrGet(MlirContext mlirCtx, std::vector<int64_t> vectorized_axis,
                                         std::vector<MlirAttribute> vectorized_strides, int64_t tensor_id,
                                         int64_t reuse_id, MlirAttribute position_config, int64_t position_id) {
  std::vector<mlir::Attribute> vectorizedStrides;
  for (auto stride : vectorized_strides) {
    vectorizedStrides.push_back(unwrap(stride));
  }
  return wrap(mlir::afir::AscTensorGroupsAttr::get(
      unwrap(mlirCtx), vectorized_axis, vectorizedStrides, tensor_id, reuse_id,
      llvm::cast<mlir::afir::PositionConfigAttr>(unwrap(position_config)), position_id));
}

std::vector<int64_t> mlirAscTensorGroupsAttrGetVectorizeAxis(MlirAttribute attr) {
  auto ascTensorGroupsAttr = llvm::cast<mlir::afir::AscTensorGroupsAttr>(unwrap(attr));
  return ascTensorGroupsAttr.getVectorizedAxis();
}

std::vector<MlirAttribute> mlirAscTensorGroupsAttrGetVectorizeStrides(MlirAttribute attr) {
  auto ascTensorGroupsAttr = llvm::cast<mlir::afir::AscTensorGroupsAttr>(unwrap(attr));
  std::vector<MlirAttribute> res;
  for (auto stride : ascTensorGroupsAttr.getVectorizedStrides()) {
    res.push_back(wrap(stride));
  }
  return res;
}

int64_t mlirAscTensorGroupsAttrGetTensorId(MlirAttribute attr) {
  auto ascTensorGroupsAttr = llvm::cast<mlir::afir::AscTensorGroupsAttr>(unwrap(attr));
  return ascTensorGroupsAttr.getTensorId();
}

int64_t mlirAscTensorGroupsAttrGetReuseId(MlirAttribute attr) {
  auto ascTensorGroupsAttr = llvm::cast<mlir::afir::AscTensorGroupsAttr>(unwrap(attr));
  return ascTensorGroupsAttr.getReuseId();
}

MlirAttribute mlirAscTensorGroupsAttrGePositionConfig(MlirAttribute attr) {
  auto ascTensorGroupsAttr = llvm::cast<mlir::afir::AscTensorGroupsAttr>(unwrap(attr));
  return wrap(ascTensorGroupsAttr.getPositionConfig());
}

int64_t mlirAscTensorGroupsAttrGetPositionId(MlirAttribute attr) {
  auto ascTensorGroupsAttr = llvm::cast<mlir::afir::AscTensorGroupsAttr>(unwrap(attr));
  return ascTensorGroupsAttr.getPositionId();
}

bool mlirAttributeIsAFIRAxisAttr(MlirAttribute attr) {
  return llvm::isa<mlir::afir::AxisAttr>(unwrap(attr));
}

MlirAttribute mlirAxisAttrGet(MlirContext mlirCtx, int64_t id, MlirAttribute name, mlir::afir::AxisType axisType,
                              bool bind_block, MlirAttribute size, MlirAttribute align, std::vector<int64_t> from) {
  return wrap(mlir::afir::AxisAttr::get(unwrap(mlirCtx), id, llvm::cast<mlir::StringAttr>(unwrap(name)), axisType,
                                        bind_block, llvm::cast<mlir::StringAttr>(unwrap(size)),
                                        llvm::cast<mlir::StringAttr>(unwrap(align)), from));
}

int64_t mlirAxisAttrGetId(MlirAttribute attr) {
  auto axisAttr = llvm::cast<mlir::afir::AxisAttr>(unwrap(attr));
  return axisAttr.getId();
}

MlirAttribute mlirAxisAttrGetName(MlirAttribute attr) {
  auto axisAttr = llvm::cast<mlir::afir::AxisAttr>(unwrap(attr));
  return wrap(llvm::cast<mlir::Attribute>(axisAttr.getName()));
}

mlir::afir::AxisType mlirAxisAttrGetAxisType(MlirAttribute attr) {
  auto axisAttr = llvm::cast<mlir::afir::AxisAttr>(unwrap(attr));
  return axisAttr.getAxisType();
}

bool mlirAxisAttrGetBindBlock(MlirAttribute attr) {
  auto axisAttr = llvm::cast<mlir::afir::AxisAttr>(unwrap(attr));
  return axisAttr.getBindBlock();
}

MlirAttribute mlirAxisAttrGetSize(MlirAttribute attr) {
  auto axisAttr = llvm::cast<mlir::afir::AxisAttr>(unwrap(attr));
  return wrap(llvm::cast<mlir::Attribute>(axisAttr.getSize()));
}

MlirAttribute mlirAxisAttrGetAlign(MlirAttribute attr) {
  mlir::afir::AxisAttr axisAttr = llvm::cast<mlir::afir::AxisAttr>(unwrap(attr));
  return wrap(llvm::cast<mlir::Attribute>(axisAttr.getAlign()));
}

std::vector<int64_t> mlirAxisAttrGetFrom(MlirAttribute attr) {
  auto axisAttr = llvm::cast<mlir::afir::AxisAttr>(unwrap(attr));
  return axisAttr.getFrom();
}

bool mlirAttributeIsAscGraphAttrGroupsAttr(MlirAttribute attr) {
  return llvm::isa<mlir::afir::AscGraphAttrGroupsAttr>(unwrap(attr));
}

MlirAttribute mlirAscGraphAttrGroupsAttrGet(MlirContext mlirCtx, int64_t tiling_key, std::vector<MlirAttribute> axes,
                                            mlir::afir::AscGraphType type, std::vector<MlirAttribute> size_var) {
  std::vector<mlir::afir::AxisAttr> realAxes;
  for (auto ax : axes) {
    realAxes.push_back(llvm::cast<mlir::afir::AxisAttr>(unwrap(ax)));
  }
  std::vector<mlir::StringAttr> realSizeVar;
  for (auto var : size_var) {
    realSizeVar.push_back(llvm::cast<mlir::StringAttr>(unwrap(var)));
  }
  return wrap(mlir::afir::AscGraphAttrGroupsAttr::get(unwrap(mlirCtx), tiling_key, realAxes, type, realSizeVar));
}