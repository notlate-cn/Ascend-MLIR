/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <optional>
#include <vector>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/nanobind.h"
#include "CAPI/Dialects.h"
#include "CAPI/Passes.h"

namespace nb = nanobind;
using namespace nanobind::literals;

NB_MODULE(_afir, m) {
  //
  // Dialects.
  //

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__afir__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);
  //
  // Attributes
  //
  nb::enum_<mlir::afir::Position>(m, "Position")
      .value("GM", mlir::afir::Position::GM)
      .value("VECTOR_IN", mlir::afir::Position::VECTOR_IN)
      .value("VECTOR_OUT", mlir::afir::Position::VECTOR_OUT)
      .value("VECTOR_CALC", mlir::afir::Position::VECTOR_CALC)
      .value("L1", mlir::afir::Position::L1)
      .value("L2", mlir::afir::Position::L2)
      .value("L0A", mlir::afir::Position::L0A)
      .value("L0B", mlir::afir::Position::L0B)
      .value("L0C", mlir::afir::Position::L0C);
  mlir::python::nanobind_adaptors::mlir_attribute_subclass(m, "PositionConfigAttr",
                                                           mlirAttributeIsAFIRPositionConfigAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, mlir::afir::Position position, uint32_t depth, bool is_double_buffer,
             MlirContext mlirCtx) {
            return cls(mlirPositionConfigAttrGet(mlirCtx, position, depth, is_double_buffer));
          },
          "cls"_a, "position"_a, "depth"_a, "is_double_buffer"_a, "mlirCtx"_a, "Get a afir.pos_config")
      .def_property_readonly("position", [](MlirAttribute self) { return mlirPositionConfigAttrGetPosition(self); })
      .def_property_readonly("depth", [](MlirAttribute self) { return mlirPositionConfigAttrGetDepth(self); })
      .def_property_readonly("is_double_buffer",
                             [](MlirAttribute self) { return mlirPositionConfigAttrGetIsDoubleBuffer(self); });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(m, "AscTensorGroupsAttr",
                                                           mlirAttributeIsAFIRAscTensorGroupsAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, std::optional<std::vector<int64_t>> vectorized_axis,
             std::optional<std::vector<MlirAttribute>> vectorized_strides, int64_t tensor_id, int64_t reuse_id,
             MlirAttribute position_config, int64_t position_id, MlirContext mlirCtx) {
            return cls(
                mlirAscTensorGroupsAttrGet(mlirCtx, vectorized_axis ? *vectorized_axis : std::vector<int64_t>(),
                                           vectorized_strides ? *vectorized_strides : std::vector<MlirAttribute>(),
                                           tensor_id, reuse_id, position_config, position_id));
          },
          "cls"_a, "vectorized_axis"_a, "vectorized_strides"_a, "tensor_id"_a, "reuse_id"_a, "position_config"_a,
          "position_id"_a, "mlirCtx"_a, "Get a afir.AscTensorGroupsAttr")
      .def_property_readonly("vectorized_axis",
                             [](MlirAttribute self) { return mlirAscTensorGroupsAttrGetVectorizeAxis(self); })
      .def_property_readonly("vectorized_strides",
                             [](MlirAttribute self) { return mlirAscTensorGroupsAttrGetVectorizeStrides(self); })
      .def_property_readonly("tensor_id", [](MlirAttribute self) { return mlirAscTensorGroupsAttrGetTensorId(self); })
      .def_property_readonly("reuse_id", [](MlirAttribute self) { return mlirAscTensorGroupsAttrGetReuseId(self); })
      .def_property_readonly("position_config",
                             [](MlirAttribute self) { return mlirAscTensorGroupsAttrGePositionConfig(self); })
      .def_property_readonly("position_id",
                             [](MlirAttribute self) { return mlirAscTensorGroupsAttrGetPositionId(self); });

  nb::enum_<mlir::afir::AxisType>(m, "AxisType")
      .value("Original", mlir::afir::AxisType::Original)
      .value("BlockOuter", mlir::afir::AxisType::BlockOuter)
      .value("BlockInner", mlir::afir::AxisType::BlockInner)
      .value("TileOuter", mlir::afir::AxisType::TileOuter)
      .value("TileInner", mlir::afir::AxisType::TileInner)
      .value("Merged", mlir::afir::AxisType::Merged)
      .value("Invalid", mlir::afir::AxisType::Invalid);

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(m, "AxisAttr", mlirAttributeIsAFIRAxisAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, int64_t id, MlirAttribute name, mlir::afir::AxisType axisType, bool bind_block,
             MlirAttribute size, MlirAttribute align, std::vector<int64_t> from, MlirContext mlirCtx) {
            return cls(mlirAxisAttrGet(mlirCtx, id, name, axisType, bind_block, size, align, from));
          },
          "cls"_a, "id"_a, "name"_a, "axisType"_a, "bind_block"_a, "size"_a, "align"_a, "from"_a, "mlirCtx"_a,
          "Get a afir.AxisAttr")
      .def_property_readonly("id", [](MlirAttribute self) { return mlirAxisAttrGetId(self); })
      .def_property_readonly("name", [](MlirAttribute self) { return mlirAxisAttrGetName(self); })
      .def_property_readonly("axisType", [](MlirAttribute self) { return mlirAxisAttrGetAxisType(self); })
      .def_property_readonly("bind_block", [](MlirAttribute self) { return mlirAxisAttrGetBindBlock(self); })
      .def_property_readonly("size", [](MlirAttribute self) { return mlirAxisAttrGetSize(self); })
      .def_property_readonly("align", [](MlirAttribute self) { return mlirAxisAttrGetAlign(self); })
      .def_property_readonly("from", [](MlirAttribute self) { return mlirAxisAttrGetFrom(self); });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(m, "AscGraphAttrGroupsAttr",
                                                           mlirAttributeIsAscGraphAttrGroupsAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, int64_t tiling_key, std::vector<MlirAttribute> axes, mlir::afir::AscGraphType type,
             std::optional<std::vector<MlirAttribute>> size_var, MlirContext mlirCtx) {
            return cls(mlirAscGraphAttrGroupsAttrGet(mlirCtx, tiling_key, axes, type,
                                                     size_var ? *size_var : std::vector<MlirAttribute>()));
          },
          "cls"_a, "tiling_key"_a, "axes"_a, "type"_a, "size_var"_a, "mlirCtx"_a, "Get a afir.AscGraphAttrGroupsAttr");

  //
  // Passes.
  //

  m.def("register_afir_passes", []() { mlirRegisterAFIRPasses(); });

  //
  // Types.
  //
}
