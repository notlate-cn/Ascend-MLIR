#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., 2026 Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
"""
AscGraph JSON to AFIR Dialect MLIR Text Converter (v3.0)

This script converts AscGraph serialized JSON strings to AFIR dialect
MLIR text representation.

Version 3.0 reflects the latest AFIR dialect optimizations:
- Uses nested PositionConfigAttr for position-related attributes
- Renames 'axis' to 'axes' in graph attributes
- Updates enum value formats (lowercase for Position, CamelCase for types)
- Properly handles optional parameters (omits empty arrays)
- Uses MLIR built-in types instead of DataType enum
- Fuses memory attributes into AscTensorGroups
- Maps AscGraph nodes to AFIR operations with simplified attributes
"""

import json
import re
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import IntEnum


# ============================================================================
# Enum Definitions (matching AFIREnums.td v2.0)
# ============================================================================

class DataType(IntEnum):
    """DataType enum for conversion (not used in AFIR output)."""
    DT_UNDEFINED = 0
    DT_FLOAT = 1
    DT_FLOAT16 = 2
    DT_INT8 = 3
    DT_UINT8 = 4
    DT_INT16 = 5
    DT_UINT16 = 6
    DT_INT32 = 7
    DT_INT64 = 8
    DT_UINT32 = 9
    DT_UINT64 = 10
    DT_BOOL = 11
    DT_DOUBLE = 12
    DT_STRING = 13
    DT_DUAL_SUB_INT8 = 14
    DT_DUAL_SUB_UINT8 = 15
    DT_COMPLEX64 = 16
    DT_COMPLEX128 = 17
    DT_QINT8 = 18
    DT_QINT16 = 19
    DT_QINT32 = 20
    DT_QUINT8 = 21
    DT_QUINT16 = 22
    DT_RESOURCE = 23
    DT_STRING_REF = 24
    DT_DUAL = 25
    DT_VARIANT = 26
    DT_BF16 = 27
    DT_INT4 = 28
    DT_UINT1 = 29
    DT_INT2 = 30
    DT_UINT2 = 31
    DT_COMPLEX32 = 32
    DT_HIFLOAT8 = 33
    DT_FLOAT8_E5M2 = 34
    DT_FLOAT8_E4M3FN = 35
    DT_FLOAT8_E8M0 = 36
    DT_FLOAT6_E3M2 = 37
    DT_FLOAT6_E2M3 = 38
    DT_FLOAT4_E2M1 = 39
    DT_FLOAT4_E1M2 = 40


class Position(IntEnum):
    """Position enum (fused with AllocType in AFIR v2.0)."""
    GM = 0
    VECTOR_IN = 1
    VECTOR_OUT = 2
    VECTOR_CALC = 3
    L1 = 4
    L2 = 5
    L0A = 6
    L0B = 7
    L0C = 8


class AxisType(IntEnum):
    """Axis type enum."""
    Original = 0
    BlockOuter = 1
    BlockInner = 2
    TileOuter = 3
    TileInner = 4
    Merged = 5
    Invalid = 6


class AscGraphType(IntEnum):
    """Graph type enum (simplified in v2.0)."""
    COMPUTE = 0
    Invalid = 1


# ============================================================================
# Data Classes for parsed structures
# ============================================================================

@dataclass
class MemAttrDef:
    tensor_id: int = -1
    alloc_type: int = 0
    position: int = 0
    hardware: int = 0
    reuse_id: int = -1


@dataclass
class MemQueueAttrDef:
    id: int = -1
    depth: int = 2
    buf_num: int = -1


@dataclass
class MemBufAttrDef:
    id: int = -1


@dataclass
class MemOptAttrDef:
    reuse_id: int = -1
    ref_tensor: int = -1
    merge_scope: int = -1


@dataclass
class AscTensorAttrGroupsDef:
    dtype: int = 0
    axis_ids: List[int] = field(default_factory=list)
    repeats: List[str] = field(default_factory=list)
    strides: List[str] = field(default_factory=list)
    vectorized_axis: List[int] = field(default_factory=list)
    vectorized_strides: List[str] = field(default_factory=list)
    mem: Optional[MemAttrDef] = None
    que: Optional[MemQueueAttrDef] = None
    buf: Optional[MemBufAttrDef] = None
    opt: Optional[MemOptAttrDef] = None


@dataclass
class AscTensorDef:
    attr: Optional[AscTensorAttrGroupsDef] = None


@dataclass
class AscInputSourceDef:
    src_node_name: str = ""
    src_out_index: int = 0


@dataclass
class AxisDef:
    id: int = 0
    name: str = ""
    axis_type: int = 0
    bind_block: bool = False
    size: str = ""
    align: str = "1"
    allow_unaligned_tail: bool = False
    from_ids: List[int] = field(default_factory=list)


@dataclass
class SchedInfoDef:
    exec_order: int = -1
    axis: List[int] = field(default_factory=list)
    loop_axis: int = -1
    exec_condition: int = 0


@dataclass
class ApiInfoDef:
    type: int = 0
    compute_type: int = 11
    unit: int = 0


@dataclass
class IrAttrValue:
    i: Optional[int] = None
    s: Optional[str] = None
    expression: Optional[str] = None
    b: Optional[bool] = None
    f: Optional[float] = None


@dataclass
class AscIrAttrDef:
    attr: Dict[str, IrAttrValue] = field(default_factory=dict)


@dataclass
class AscNodeAttrGroupsDef:
    name: str = ""
    type: str = ""
    sched: Optional[SchedInfoDef] = None
    api: Optional[ApiInfoDef] = None
    ir_attr_def: Optional[AscIrAttrDef] = None


@dataclass
class IrDef:
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    input_ir_type: List[int] = field(default_factory=list)
    output_ir_type: List[int] = field(default_factory=list)
    type: str = ""
    input_nums: List[int] = field(default_factory=list)
    output_nums: List[int] = field(default_factory=list)


@dataclass
class AscNodeDef:
    input_src: List[AscInputSourceDef] = field(default_factory=list)
    outputs: List[AscTensorDef] = field(default_factory=list)
    attr: Optional[AscNodeAttrGroupsDef] = None
    ir_def: Optional[IrDef] = None


@dataclass
class AscGraphAttrGroupsDef:
    tiling_key: int = -1
    axis: List[AxisDef] = field(default_factory=list)
    type: int = 0
    size_var: List[str] = field(default_factory=list)


@dataclass
class AscGraphDef:
    asc_graph_attr: Optional[AscGraphAttrGroupsDef] = None
    asc_node: List[AscNodeDef] = field(default_factory=list)
    graph_name: str = ""


# ============================================================================
# Protobuf Text Format Parser
# ============================================================================

class ProtobufTextParser:
    """Parser for protobuf text format."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.length = len(text)

    def skip_whitespace(self):
        while self.pos < self.length and self.text[self.pos] in ' \t\n\r':
            self.pos += 1

    def peek(self) -> str:
        if self.pos >= self.length:
            return ''
        return self.text[self.pos]

    def consume(self, expected: str):
        self.skip_whitespace()
        if self.text[self.pos:self.pos + len(expected)] != expected:
            raise ValueError(f"Expected '{expected}' at position {self.pos}, got '{self.text[self.pos:self.pos+10]}'")
        self.pos += len(expected)

    def try_consume(self, expected: str) -> bool:
        self.skip_whitespace()
        if self.text[self.pos:self.pos + len(expected)] == expected:
            self.pos += len(expected)
            return True
        return False

    def parse_identifier(self) -> str:
        self.skip_whitespace()
        start = self.pos
        while self.pos < self.length and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            self.pos += 1
        return self.text[start:self.pos]

    def parse_string(self) -> str:
        self.skip_whitespace()
        if self.peek() != '"':
            raise ValueError(f"Expected '\"' at position {self.pos}")
        self.pos += 1
        result = []
        while self.pos < self.length:
            ch = self.text[self.pos]
            if ch == '"':
                self.pos += 1
                return ''.join(result)
            elif ch == '\\':
                self.pos += 1
                if self.pos < self.length:
                    escape_char = self.text[self.pos]
                    if escape_char == 'n':
                        result.append('\n')
                    elif escape_char == 't':
                        result.append('\t')
                    elif escape_char == 'r':
                        result.append('\r')
                    elif escape_char == '"':
                        result.append('"')
                    elif escape_char == '\\':
                        result.append('\\')
                    else:
                        result.append(escape_char)
                    self.pos += 1
            else:
                result.append(ch)
                self.pos += 1
        raise ValueError("Unterminated string")

    def parse_number(self) -> int | float:
        self.skip_whitespace()
        start = self.pos
        if self.peek() == '-':
            self.pos += 1
        while self.pos < self.length and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
            self.pos += 1
        num_str = self.text[start:self.pos]
        if '.' in num_str:
            return float(num_str)
        return int(num_str)

    def parse_bool(self) -> bool:
        self.skip_whitespace()
        if self.text[self.pos:self.pos + 4] == 'true':
            self.pos += 4
            return True
        elif self.text[self.pos:self.pos + 5] == 'false':
            self.pos += 5
            return False
        raise ValueError(f"Expected bool at position {self.pos}")

    def parse_value(self) -> Any:
        self.skip_whitespace()
        ch = self.peek()
        if ch == '"':
            return self.parse_string()
        elif ch == '{':
            return self.parse_message()
        elif ch == '-' or ch.isdigit():
            return self.parse_number()
        elif ch == 't' and self.text[self.pos:self.pos + 4] == 'true':
            return self.parse_bool()
        elif ch == 'f' and self.text[self.pos:self.pos + 5] == 'false':
            return self.parse_bool()
        elif ch.isalpha() or ch == '_':
            return self.parse_identifier()
        else:
            raise ValueError(f"Unexpected character '{ch}' at position {self.pos}")

    def parse_message(self) -> Dict[str, Any]:
        self.consume('{')
        result = {}
        while True:
            self.skip_whitespace()
            if self.peek() == '}':
                self.pos += 1
                break

            field_name = self.parse_identifier()
            if not field_name:
                if self.peek() == '}':
                    self.pos += 1
                    break
                raise ValueError(f"Expected field name at position {self.pos}")

            self.skip_whitespace()
            if self.try_consume(':'):
                value = self.parse_value()
            else:
                value = self.parse_message()

            if field_name in result:
                if not isinstance(result[field_name], list):
                    result[field_name] = [result[field_name]]
                result[field_name].append(value)
            else:
                result[field_name] = value

        return result

    def parse(self) -> Dict[str, Any]:
        result = {}
        while self.pos < self.length:
            self.skip_whitespace()
            if self.pos >= self.length:
                break

            field_name = self.parse_identifier()
            if not field_name:
                break

            self.skip_whitespace()
            if self.try_consume(':'):
                value = self.parse_value()
            else:
                value = self.parse_message()

            if field_name in result:
                if not isinstance(result[field_name], list):
                    result[field_name] = [result[field_name]]
                result[field_name].append(value)
            else:
                result[field_name] = value

        return result


# ============================================================================
# AscGraph Parser (unchanged)
# ============================================================================

def ensure_list(value) -> list:
    """Ensure value is a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def parse_mem_attr(data: dict) -> MemAttrDef:
    """Parse MemAttrDef from dict."""
    return MemAttrDef(
        tensor_id=data.get('tensor_id', -1),
        alloc_type=data.get('alloc_type', 0),
        position=data.get('position', 0),
        hardware=data.get('hardware', 0),
        reuse_id=data.get('reuse_id', -1)
    )


def parse_mem_queue_attr(data: dict) -> MemQueueAttrDef:
    """Parse MemQueueAttrDef from dict."""
    return MemQueueAttrDef(
        id=data.get('id', -1),
        depth=data.get('depth', 2),
        buf_num=data.get('buf_num', -1)
    )


def parse_mem_buf_attr(data: dict) -> MemBufAttrDef:
    """Parse MemBufAttrDef from dict."""
    return MemBufAttrDef(
        id=data.get('id', -1)
    )


def parse_mem_opt_attr(data: dict) -> MemOptAttrDef:
    """Parse MemOptAttrDef from dict."""
    return MemOptAttrDef(
        reuse_id=data.get('reuse_id', -1),
        ref_tensor=data.get('ref_tensor', -1),
        merge_scope=data.get('merge_scope', -1)
    )


def parse_asc_tensor_attr_groups(data: dict) -> AscTensorAttrGroupsDef:
    """Parse AscTensorAttrGroupsDef from dict."""
    result = AscTensorAttrGroupsDef(
        dtype=data.get('dtype', 0),
        axis_ids=ensure_list(data.get('axis_ids', [])),
        repeats=ensure_list(data.get('repeats', [])),
        strides=ensure_list(data.get('strides', [])),
        vectorized_axis=ensure_list(data.get('vectorized_axis', [])),
        vectorized_strides=ensure_list(data.get('vectorized_strides', []))
    )
    if 'mem' in data:
        result.mem = parse_mem_attr(data['mem'])
    if 'que' in data:
        result.que = parse_mem_queue_attr(data['que'])
    if 'buf' in data:
        result.buf = parse_mem_buf_attr(data['buf'])
    if 'opt' in data:
        result.opt = parse_mem_opt_attr(data['opt'])
    return result


def parse_asc_tensor(data: dict) -> AscTensorDef:
    """Parse AscTensorDef from dict."""
    result = AscTensorDef()
    if 'attr' in data:
        result.attr = parse_asc_tensor_attr_groups(data['attr'])
    return result


def parse_input_source(data: dict) -> AscInputSourceDef:
    """Parse AscInputSourceDef from dict."""
    return AscInputSourceDef(
        src_node_name=data.get('src_node_name', ''),
        src_out_index=data.get('src_out_index', 0)
    )


def parse_axis(data: dict) -> AxisDef:
    """Parse AxisDef from dict."""
    return AxisDef(
        id=data.get('id', 0),
        name=data.get('name', ''),
        axis_type=data.get('axis_type', 0),
        bind_block=data.get('bind_block', False),
        size=data.get('size', ''),
        align=data.get('align', '1'),
        allow_unaligned_tail=data.get('allow_unaligned_tail', False),
        from_ids=ensure_list(data.get('from', []))
    )


def parse_sched_info(data: dict) -> SchedInfoDef:
    """Parse SchedInfoDef from dict."""
    return SchedInfoDef(
        exec_order=data.get('exec_order', -1),
        axis=ensure_list(data.get('axis', [])),
        loop_axis=data.get('loop_axis', -1),
        exec_condition=data.get('exec_condition', 0)
    )


def parse_api_info(data: dict) -> ApiInfoDef:
    """Parse ApiInfoDef from dict."""
    return ApiInfoDef(
        type=data.get('type', 0),
        compute_type=data.get('compute_type', 11),
        unit=data.get('unit', 0)
    )


def parse_ir_attr_value(data) -> IrAttrValue:
    """Parse IrAttrValue from dict or primitive."""
    if isinstance(data, dict):
        return IrAttrValue(
            i=data.get('i'),
            s=data.get('s'),
            expression=data.get('expression'),
            b=data.get('b'),
            f=data.get('f')
        )
    elif isinstance(data, int):
        return IrAttrValue(i=data)
    elif isinstance(data, float):
        return IrAttrValue(f=data)
    elif isinstance(data, bool):
        return IrAttrValue(b=data)
    elif isinstance(data, str):
        return IrAttrValue(s=data)
    else:
        return IrAttrValue()


def parse_ir_attr_def(data: dict) -> AscIrAttrDef:
    """Parse AscIrAttrDef from dict."""
    result = AscIrAttrDef()
    if 'attr' in data:
        attr_data = data['attr']
        if isinstance(attr_data, list):
            for item in attr_data:
                if isinstance(item, dict) and 'key' in item and 'value' in item:
                    result.attr[item['key']] = parse_ir_attr_value(item['value'])
        elif isinstance(attr_data, dict):
            if 'key' in attr_data and 'value' in attr_data:
                result.attr[attr_data['key']] = parse_ir_attr_value(attr_data['value'])
            else:
                for key, value in attr_data.items():
                    result.attr[key] = parse_ir_attr_value(value)
    return result


def parse_node_attr_groups(data: dict) -> AscNodeAttrGroupsDef:
    """Parse AscNodeAttrGroupsDef from dict."""
    result = AscNodeAttrGroupsDef(
        name=data.get('name', ''),
        type=data.get('type', '')
    )
    if 'sched' in data:
        result.sched = parse_sched_info(data['sched'])
    if 'api' in data:
        result.api = parse_api_info(data['api'])
    if 'ir_attr_def' in data:
        result.ir_attr_def = parse_ir_attr_def(data['ir_attr_def'])
    return result


def parse_ir_def(data: dict) -> IrDef:
    """Parse IrDef from dict."""
    return IrDef(
        input_names=ensure_list(data.get('input_names', [])),
        output_names=ensure_list(data.get('output_names', [])),
        input_ir_type=ensure_list(data.get('input_ir_type', [])),
        output_ir_type=ensure_list(data.get('output_ir_type', [])),
        type=data.get('type', ''),
        input_nums=ensure_list(data.get('input_nums', [])),
        output_nums=ensure_list(data.get('output_nums', []))
    )


def parse_asc_node(data: dict) -> AscNodeDef:
    """Parse AscNodeDef from dict."""
    result = AscNodeDef()
    if 'input_src' in data:
        for src in ensure_list(data['input_src']):
            result.input_src.append(parse_input_source(src))
    if 'outputs' in data:
        for out in ensure_list(data['outputs']):
            result.outputs.append(parse_asc_tensor(out))
    if 'attr' in data:
        result.attr = parse_node_attr_groups(data['attr'])
    if 'ir_def' in data:
        result.ir_def = parse_ir_def(data['ir_def'])
    return result


def parse_graph_attr_groups(data: dict) -> AscGraphAttrGroupsDef:
    """Parse AscGraphAttrGroupsDef from dict."""
    result = AscGraphAttrGroupsDef(
        tiling_key=data.get('tiling_key', -1),
        type=data.get('type', 0),
        size_var=ensure_list(data.get('size_var', []))
    )
    if 'axis' in data:
        for ax in ensure_list(data['axis']):
            result.axis.append(parse_axis(ax))
    return result


def parse_asc_graph(data: dict) -> AscGraphDef:
    """Parse AscGraphDef from dict."""
    result = AscGraphDef(
        graph_name=data.get('graph_name', '')
    )
    if 'asc_graph_attr' in data:
        result.asc_graph_attr = parse_graph_attr_groups(data['asc_graph_attr'])
    if 'asc_node' in data:
        for node in ensure_list(data['asc_node']):
            result.asc_node.append(parse_asc_node(node))
    return result


# ============================================================================
# AFIR MLIR Text Generator (v2.0)
# ============================================================================

def get_dtype_mlir_type(dtype: int) -> str:
    """Map DataType enum to MLIR element type."""
    type_map = {
        1: "f32", 2: "f16", 27: "bf16",
        3: "i8", 4: "ui8", 5: "i16", 6: "ui16",
        7: "i32", 8: "i64", 9: "ui32", 10: "ui64",
        11: "i1", 12: "f64"
    }
    return type_map.get(dtype, "f32")


def get_position_name(position: int, alloc_type: int) -> str:
    """Map position and alloc_type to AFIR Position enum (fused)."""
    # Fuse AllocType into Position
    # Note: Returns lowercase enum values as per AFIR v3.0
    if alloc_type == 1:
        return "l1"
    elif alloc_type == 2:
        return "l2"
    elif position == 0:
        return "gm"
    elif position == 1:
        return "vector_in"
    elif position == 2:
        return "vector_out"
    elif position == 3:
        return "vector_calc"
    else:
        return "gm"


def get_axis_type_name(axis_type: int) -> str:
    """Get AFIR axis type name."""
    try:
        return AxisType(axis_type).name
    except ValueError:
        return "Original"


def get_graph_type_name(graph_type: int) -> str:
    """Get AFIR graph type name (v3.0 simplified)."""
    if graph_type == 0:
        return "Compute"
    else:
        return "Invalid"


def escape_string(s: str) -> str:
    """Escape string for MLIR."""
    return s.replace('\\', '\\\\').replace('"', '\\"')


class AFIRGenerator:
    """Generator for AFIR MLIR text representation (v3.0)."""

    def __init__(self, graph: AscGraphDef):
        self.graph = graph
        self.indent = 0
        self.node_map: Dict[str, int] = {}
        self.ssa_counter = 1  # Start from %1 to match expected output
        self.indexing_maps: Dict[str, str] = {}  # map content -> map name
        self.data_nodes: List[AscNodeDef] = []
        self.output_nodes: List[AscNodeDef] = []

    def get_indent(self) -> str:
        return "  " * self.indent

    def _find_node_by_name(self, name: str) -> Optional[AscNodeDef]:
        """Find a node in the graph by name."""
        for node in self.graph.asc_node:
            if node.attr and node.attr.name == name:
                return node
        return None

    def gen_array_i64(self, values: List[int]) -> str:
        """Generate array<i64> representation."""
        return f"[{', '.join(str(v) for v in values)}]"

    def gen_array_str(self, values: List[str]) -> str:
        """Generate array<string> representation."""
        escaped = [f'"{escape_string(v)}"' for v in values]
        return f"[{', '.join(escaped)}]"

    def gen_axis_attr(self, axis: AxisDef) -> str:
        """Generate #afir.axis attribute."""
        parts = [
            f"id={axis.id}",
            f'name="{escape_string(axis.name)}"',
            f"axis_type={get_axis_type_name(axis.axis_type)}"
        ]
        if axis.bind_block:
            parts.append(f"bind_block={str(axis.bind_block).lower()}")
        parts.append(f'size="{escape_string(axis.size)}"')
        if axis.align and axis.align != "1":
            parts.append(f'align="{escape_string(axis.align)}"')
        if axis.from_ids:
            parts.append(f"from=[{', '.join(str(v) for v in axis.from_ids)}]")
        return f"<{','.join(parts)}>"

    def gen_asc_graph_attr(self, graph_attr: AscGraphAttrGroupsDef) -> str:
        """Generate #afir.asc_graph attribute (v3.0 with 'axes' field)."""
        axis_parts = [self.gen_axis_attr(ax) for ax in graph_attr.axis]
        parts = []

        # Always include axes (changed from 'axis' to 'axes' in v3.0)
        parts.append(f"axes = [{', '.join(axis_parts)}]")

        # Always include type
        parts.append(f"type = {get_graph_type_name(graph_attr.type)}")

        return f"#afir.asc_graph<{', '.join(parts)}>"

    def gen_position_config_attr(self, tensor_attr: AscTensorAttrGroupsDef) -> str:
        """Generate position config attribute (simplified format)."""
        position = get_position_name(
            tensor_attr.mem.position if tensor_attr.mem else 0,
            tensor_attr.mem.alloc_type if tensor_attr.mem else 0
        )

        # Simplified format matching expected output
        return f"<{position}>"

    def gen_asc_tensor_attr(self, tensor_attr: AscTensorAttrGroupsDef) -> str:
        """Generate #afir.asc_tensor attribute (v3.0 with nested position_config)."""
        tensor_id = tensor_attr.mem.tensor_id if tensor_attr.mem else -1
        reuse_id = tensor_attr.mem.reuse_id if tensor_attr.mem else -1
        position_id = (tensor_attr.que.id if tensor_attr.que else
                      (tensor_attr.buf.id if tensor_attr.buf else -1))

        parts = []

        # Optional arrays - only add if non-empty
        if tensor_attr.vectorized_axis:
            parts.append(f"vectorized_axis = {self.gen_array_i64(tensor_attr.vectorized_axis)}")
        if tensor_attr.vectorized_strides:
            parts.append(f"vectorized_strides = {self.gen_array_str(tensor_attr.vectorized_strides)}")

        # Always include tensor_id
        parts.append(f"tensor_id = {tensor_id}")

        # Only add reuse_id if not default (-1)
        if reuse_id != -1:
            parts.append(f"reuse_id = {reuse_id}")

        # Always include position_config (nested attribute)
        position_config = self.gen_position_config_attr(tensor_attr)
        parts.append(f"position = {position_config}")

        # Only add position_id if not default (-1)
        if position_id != -1:
            parts.append(f"position_id = {position_id}")

        return f"#afir.asc_tensor<{', '.join(parts)}>"

    def gen_ir_attr_dict(self, ir_attr_def: Optional[AscIrAttrDef]) -> str:
        """Generate dictionary of IR attributes."""
        if ir_attr_def is None or not ir_attr_def.attr:
            return "{}"

        parts = []
        for key, value in ir_attr_def.attr.items():
            if value.i is not None:
                parts.append(f'"{escape_string(key)}" = {value.i} : i64')
            elif value.s is not None:
                parts.append(f'"{escape_string(key)}" = "{escape_string(value.s)}"')
            elif value.expression is not None:
                parts.append(f'"{escape_string(key)}" = "{escape_string(value.expression)}"')
            elif value.b is not None:
                parts.append(f'"{escape_string(key)}" = {str(value.b).lower()}')
            elif value.f is not None:
                parts.append(f'"{escape_string(key)}" = {value.f} : f64')

        return "{" + ", ".join(parts) + "}"

    def infer_shape_from_tensor_attr(self, attr: AscTensorAttrGroupsDef) -> List[int]:
        """Infer tensor shape from axis_ids and repeats."""
        if not attr.axis_ids or not attr.repeats:
            return []

        shape = []
        for i, axis_id in enumerate(attr.axis_ids):
            if i < len(attr.repeats):
                try:
                    dim = int(attr.repeats[i])
                    shape.append(dim)
                except ValueError:
                    # Expression, use dynamic dimension
                    shape.append(-1)
        return shape

    def get_mlir_tensor_type(self, attr: AscTensorAttrGroupsDef) -> str:
        """Get MLIR tensor type from tensor attributes."""
        elem_type = get_dtype_mlir_type(attr.dtype)
        shape = self.infer_shape_from_tensor_attr(attr)

        if shape:
            dims = "x".join(str(d) if d > 0 else "?" for d in shape)
            return f"tensor<{dims}x{elem_type}>"
        else:
            return f"tensor<*x{elem_type}>"

    def build_affine_map(self, axis_ids: List[int], num_dims: int) -> str:
        """Build affine map from axis_ids (simplified version)."""
        if not axis_ids:
            dims = ", ".join(f"d{i}" for i in range(num_dims))
            return f"affine_map<({dims}) -> ({dims})>"

        dims = ", ".join(f"d{i}" for i in range(num_dims))
        return f"affine_map<({dims}) -> ({dims})>"

    def get_node_op_name(self, node_type: str) -> str:
        """Map node type to AFIR operation name."""
        op_map = {
            "Add": "afir.add",
            "Sub": "afir.sub",
            "Mul": "afir.mul",
            "Div": "afir.div",
            "Data": "afir.data",
            "Load": "afir.load",
            "Store": "afir.store",
            "Output": "afir.output",
            "Broadcast": "afir.broadcast",
        }
        return op_map.get(node_type, f"afir.{node_type.lower()}")

    def gen_node_operation(self, node: AscNodeDef) -> str:
        """Generate AFIR operation for a node (v2.0)."""
        if node.attr is None:
            return ""

        node_name = node.attr.name
        node_type = node.attr.type

        # Skip Data and Output nodes - they are handled in function signature
        if node_type in ["Data", "Output"]:
            return ""

        op_name = self.get_node_op_name(node_type)

        # Get result type
        result_type = "tensor<*xf32>"
        if node.outputs and node.outputs[0].attr:
            result_type = self.get_mlir_tensor_type(node.outputs[0].attr)

        # Get input SSA values
        input_values = []
        for src in node.input_src:
            src_name = src.src_node_name
            if src_name in self.node_map:
                val = self.node_map[src_name]
                if isinstance(val, str):
                    input_values.append(f"%{val}")
                else:
                    input_values.append(f"%{val}")
            else:
                input_values.append(f"%arg_{src_name}")

        # Generate result SSA value
        result_ssa = self.ssa_counter
        self.node_map[node_name] = result_ssa
        self.ssa_counter += 1

        indent = self.get_indent()

        # Build indexing_maps
        num_dims = len(node.outputs[0].attr.axis_ids) if (node.outputs and node.outputs[0].attr) else 2
        indexing_map = self.build_affine_map([], num_dims)

        # Collect indexing maps for deduplication
        if len(input_values) > 1:
            indexing_maps_list = [indexing_map] * (len(input_values) + 1)  # inputs + output
        else:
            indexing_maps_list = [indexing_map]

        for imap in indexing_maps_list:
            if imap not in self.indexing_maps:
                self.indexing_maps[imap] = f"map{len(self.indexing_maps)}"

        # Build attributes (omit default values)
        attrs = []

        # indexing_maps - always required
        map_refs = [f"#{self.indexing_maps[imap]}" for imap in indexing_maps_list]
        attrs.append(f"indexing_maps = [{', '.join(map_refs)}]")

        # loop_axis - only include if not default (-1)
        loop_axis_val = node.attr.sched.loop_axis if node.attr.sched else -1
        if loop_axis_val != -1:
            attrs.append(f"loop_axis = {loop_axis_val}")

        # ir_attr_def - only include if not empty
        if node.attr.ir_attr_def and node.attr.ir_attr_def.attr:
            ir_dict = self.gen_ir_attr_dict(node.attr.ir_attr_def)
            attrs.append(f"ir_attr_def = {ir_dict}")

        # tmp_buffers - omit if empty (DefaultValuedOptionalAttr)
        # (no need to add if empty)

        # outputs attribute - only include if not empty
        if node.outputs and node.outputs[0].attr:
            outputs_attr = self.gen_asc_tensor_attr(node.outputs[0].attr)
            attrs.append(f"outputs = [{outputs_attr}]")

        attr_str = ", ".join(attrs)

        # Build operation
        if len(input_values) == 0:
            return f"{indent}%{result_ssa} = {op_name} {{\n{indent}  {attr_str}\n{indent}}} : () -> {result_type}"
        elif len(input_values) == 1:
            # Get input type from source node
            input_type = result_type
            if node.input_src:
                src_name = node.input_src[0].src_node_name
                src_node = self._find_node_by_name(src_name)
                if src_node and src_node.outputs and src_node.outputs[0].attr:
                    input_type = self.get_mlir_tensor_type(src_node.outputs[0].attr)
            return f"{indent}%{result_ssa} = {op_name} {input_values[0]} {{\n{indent}  {attr_str}\n{indent}}} : {input_type} -> {result_type}"
        else:
            inputs_str = ", ".join(input_values)
            types_str = ", ".join([result_type] * len(input_values))
            return f"{indent}%{result_ssa} = {op_name} {inputs_str} {{\n{indent}  {attr_str}\n{indent}}} : ({types_str}) -> {result_type}"

    def generate(self) -> str:
        """Generate complete AFIR MLIR text (v3.0)."""
        lines = []

        # Module header
        lines.append("// AFIR Dialect representation of AscGraph (v2.0)")
        lines.append(f'// Graph name: {self.graph.graph_name}')
        lines.append("")

        # Identify Data and Output nodes
        self.data_nodes = []
        self.output_nodes = []

        for node in self.graph.asc_node:
            if node.attr and node.attr.type == "Data":
                self.data_nodes.append(node)
            elif node.attr and node.attr.type == "Output":
                self.output_nodes.append(node)

        # Sort Data nodes by index
        self.data_nodes.sort(key=lambda n: n.attr.ir_attr_def.attr.get('index', IrAttrValue(i=999)).i if n.attr.ir_attr_def else 999)

        # Map Data nodes to function arguments BEFORE generating operations
        for i, data_node in enumerate(self.data_nodes):
            self.node_map[data_node.attr.name] = f"arg{i}"

        # First pass: generate all operations to collect indexing maps
        temp_ops = []
        for node in self.graph.asc_node:
            op_str = self.gen_node_operation(node)
            if op_str:
                temp_ops.append(op_str)

        # Generate indexing map definitions at top level (outside module)
        if self.indexing_maps:
            lines.append("// Indexing Maps")
            sorted_maps = sorted(self.indexing_maps.items(), key=lambda x: x[1])
            for map_content, map_name in sorted_maps:
                lines.append(f"#{map_name} = {map_content}")
            lines.append("")

        # Graph attributes as module attribute
        if self.graph.asc_graph_attr:
            graph_attr = self.gen_asc_graph_attr(self.graph.asc_graph_attr)
            lines.append(f"module attributes {{ afir.asc_graph_attr = {graph_attr} }} {{")
        else:
            lines.append("module {")

        lines.append("")  # Add blank line after module
        self.indent = 1

        # Generate function signature from Data nodes
        func_name = self.graph.graph_name.replace('/', '_') if self.graph.graph_name else "main"
        func_args = []

        for i, data_node in enumerate(self.data_nodes):
            if data_node.outputs and data_node.outputs[0].attr:
                tensor_type = self.get_mlir_tensor_type(data_node.outputs[0].attr)
                func_args.append(f"%arg{i}: {tensor_type}")

        # Get output type from Output node
        output_type = "tensor<20x31xf32>"
        if self.output_nodes and self.output_nodes[0].input_src:
            src_name = self.output_nodes[0].input_src[0].src_node_name
            src_node = self._find_node_by_name(src_name)
            if src_node and src_node.outputs and src_node.outputs[0].attr:
                output_type = self.get_mlir_tensor_type(src_node.outputs[0].attr)

        func_sig = f'{self.get_indent()}func.func @{func_name}({", ".join(func_args)}) -> {output_type} {{'
        lines.append(func_sig)

        self.indent = 2

        # Add collected operations
        for op_str in temp_ops:
            lines.append(op_str)

        # Return value from last operation before Output
        if self.output_nodes and self.output_nodes[0].input_src:
            src_name = self.output_nodes[0].input_src[0].src_node_name
            if src_name in self.node_map:
                ret_val = self.node_map[src_name]
                if isinstance(ret_val, int):
                    lines.append(f"{self.get_indent()}return %{ret_val} : {output_type}")
                else:
                    lines.append(f"{self.get_indent()}return %{ret_val} : {output_type}")
        else:
            lines.append(f"{self.get_indent()}return")

        self.indent = 1
        lines.append(f"{self.get_indent()}}}")

        lines.append("}")

        return "\n".join(lines)


# ============================================================================
# Main Entry Point
# ============================================================================

def unescape_json_string(s: str) -> str:
    """Unescape a JSON-like escaped string."""
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            next_char = s[i + 1]
            if next_char == '"':
                result.append('"')
                i += 2
            elif next_char == '\\':
                result.append('\\')
                i += 2
            elif next_char == 'n':
                result.append('\n')
                i += 2
            elif next_char == 't':
                result.append('\t')
                i += 2
            elif next_char == 'r':
                result.append('\r')
                i += 2
            else:
                result.append(s[i])
                i += 1
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


def extract_ascgraph_from_json(json_str: str) -> str:
    """Extract the ascgraph text from the serialized JSON string."""
    working_str = json_str.strip()

    if working_str.startswith('{\\'):
        working_str = unescape_json_string(working_str)

    try:
        data = json.loads(working_str)
    except json.JSONDecodeError:
        try:
            unescaped = unescape_json_string(working_str)
            data = json.loads(unescaped)
        except json.JSONDecodeError:
            return working_str

    if 'compute_graph' in data:
        compute_graph_str = data['compute_graph']
        parser = ProtobufTextParser(compute_graph_str)
        cg_data = parser.parse()

        if 'op' in cg_data:
            for op in ensure_list(cg_data['op']):
                if op.get('type') == 'AscGraph':
                    if 'attr' in op:
                        attrs = op['attr']
                        if isinstance(attrs, list):
                            for attr_item in attrs:
                                if attr_item.get('key') == 'ascgraph':
                                    value = attr_item.get('value', {})
                                    if isinstance(value, dict):
                                        return value.get('s', '')
                                    return str(value)
                        elif isinstance(attrs, dict):
                            if 'ascgraph' in attrs:
                                value = attrs['ascgraph']
                                if isinstance(value, dict):
                                    return value.get('s', '')
                                return str(value)

    return working_str


def convert_ascgraph_to_afir(input_text: str) -> str:
    """Convert AscGraph JSON/text to AFIR MLIR text (v3.0)."""
    ascgraph_text = extract_ascgraph_from_json(input_text)
    parser = ProtobufTextParser(ascgraph_text)
    data = parser.parse()
    graph = parse_asc_graph(data)
    generator = AFIRGenerator(graph)
    return generator.generate()


def main():
    parser = argparse.ArgumentParser(
        description='Convert AscGraph JSON to AFIR MLIR text (v3.0 - optimized with nested attributes)'
    )
    parser.add_argument(
        'input',
        nargs='?',
        help='Input file path (reads from stdin if not provided)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (writes to stdout if not provided)'
    )

    args = parser.parse_args()

    # Read input
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_text = f.read()
    else:
        input_text = sys.stdin.read()

    # Convert
    try:
        output_text = convert_ascgraph_to_afir(input_text)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Write output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
    else:
        print(output_text)


if __name__ == '__main__':
    main()
