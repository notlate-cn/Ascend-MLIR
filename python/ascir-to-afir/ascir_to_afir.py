#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., 2025 Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
"""
AscGraph JSON to AFIR Dialect MLIR Text Converter

This script converts AscGraph serialized JSON strings to AFIR dialect
text representation of computation graphs.
"""

import json
import re
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import IntEnum


# ============================================================================
# Enum Definitions (matching AFIREnums.td)
# ============================================================================

class DataType(IntEnum):
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


class AllocType(IntEnum):
    GLOBAL = 0
    L1 = 1
    L2 = 2
    QBUF = 3
    TBUF = 4


class Position(IntEnum):
    GM = 0
    VECTOR_IN = 1
    VECTOR_OUT = 2
    VECTOR_CALC = 3


class Hardware(IntEnum):
    GM = 0
    UB = 1


class AxisType(IntEnum):
    Original = 0
    BlockOuter = 1
    BlockInner = 2
    TileOuter = 3
    TileInner = 4
    Merged = 5
    Invalid = 6


class ExecuteCondition(IntEnum):
    NoCache = 0
    CacheBlockSplitFusedBroadcastAxis = 1
    CacheBlockSplitOriginBroadcastAxis = 2
    ConditionInvalid = 3


class ApiType(IntEnum):
    Buffer = 0
    Compute = 1
    Invalid = 2


class ComputeUnit(IntEnum):
    NONE = 0
    MTE1 = 1
    MTE2 = 2
    MTE3 = 3
    Scalar = 4
    Vector = 5
    Cube = 6
    Invalid = 7


class ComputeType(IntEnum):
    Load = 0
    Store = 1
    ReduceStore = 2
    Elewise = 3
    Broadcast = 4
    Reduce = 5
    Transpose = 6
    Concat = 7
    Gather = 8
    Cube = 9
    Split = 10
    Invalid = 11


class AscGraphType(IntEnum):
    HintGraph = 0
    ImplGraph = 1


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
    split_pair_other_id: int = -1


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
            # Handle enum values or identifiers
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

            # Handle repeated fields
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

            # Handle repeated fields
            if field_name in result:
                if not isinstance(result[field_name], list):
                    result[field_name] = [result[field_name]]
                result[field_name].append(value)
            else:
                result[field_name] = value

        return result


# ============================================================================
# AscGraph Parser
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
        from_ids=ensure_list(data.get('from', [])),
        split_pair_other_id=data.get('split_pair_other_id', -1)
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
        # Handle list of {key, value} pairs
        if isinstance(attr_data, list):
            for item in attr_data:
                if isinstance(item, dict) and 'key' in item and 'value' in item:
                    result.attr[item['key']] = parse_ir_attr_value(item['value'])
        # Handle single {key, value} pair (when there's only one attr)
        elif isinstance(attr_data, dict):
            if 'key' in attr_data and 'value' in attr_data:
                # Single {key: ..., value: ...} structure
                result.attr[attr_data['key']] = parse_ir_attr_value(attr_data['value'])
            else:
                # Direct key-value mapping
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
# AFIR MLIR Text Generator
# ============================================================================

def get_dtype_name(dtype: int) -> str:
    """Get AFIR dtype name from enum value."""
    try:
        return DataType(dtype).name
    except ValueError:
        return "DT_UNDEFINED"


def get_alloc_type_name(alloc_type: int) -> str:
    """Get AFIR alloc type name from enum value."""
    try:
        return AllocType(alloc_type).name
    except ValueError:
        return "GLOBAL"


def get_position_name(position: int) -> str:
    """Get AFIR position name from enum value."""
    try:
        return Position(position).name
    except ValueError:
        return "GM"


def get_hardware_name(hardware: int) -> str:
    """Get AFIR hardware name from enum value."""
    try:
        return Hardware(hardware).name
    except ValueError:
        return "GM"


def get_axis_type_name(axis_type: int) -> str:
    """Get AFIR axis type name from enum value."""
    try:
        return AxisType(axis_type).name
    except ValueError:
        return "Original"


def get_exec_condition_name(exec_condition: int) -> str:
    """Get AFIR execute condition name from enum value."""
    try:
        return ExecuteCondition(exec_condition).name
    except ValueError:
        return "NoCache"


def get_api_type_name(api_type: int) -> str:
    """Get AFIR api type name from enum value."""
    try:
        return ApiType(api_type).name
    except ValueError:
        return "Buffer"


def get_compute_unit_name(unit: int) -> str:
    """Get AFIR compute unit name from enum value."""
    try:
        return ComputeUnit(unit).name
    except ValueError:
        return "None"


def get_compute_type_name(compute_type: int) -> str:
    """Get AFIR compute type name from enum value."""
    try:
        return ComputeType(compute_type).name
    except ValueError:
        return "Invalid"


def get_graph_type_name(graph_type: int) -> str:
    """Get AFIR graph type name from enum value."""
    try:
        return AscGraphType(graph_type).name
    except ValueError:
        return "HintGraph"


def escape_string(s: str) -> str:
    """Escape string for MLIR."""
    return s.replace('\\', '\\\\').replace('"', '\\"')


class AFIRGenerator:
    """Generator for AFIR MLIR text representation."""

    def __init__(self, graph: AscGraphDef):
        self.graph = graph
        self.indent = 0
        self.node_map: Dict[str, int] = {}  # node_name -> SSA value index
        self.ssa_counter = 0

    def get_indent(self) -> str:
        return "  " * self.indent

    def gen_array_i64(self, values: List[int]) -> str:
        """Generate array<i64> representation."""
        return f"[{', '.join(str(v) for v in values)}]"

    def gen_array_str(self, values: List[str]) -> str:
        """Generate array<string> representation."""
        escaped = [f'"{escape_string(v)}"' for v in values]
        return f"[{', '.join(escaped)}]"

    def gen_mem_attr(self, mem: MemAttrDef) -> str:
        """Generate #afir.mem attribute."""
        parts = [
            f"tensor_id = {mem.tensor_id}",
            f"alloc_type = {get_alloc_type_name(mem.alloc_type)}",
            f"position = {get_position_name(mem.position)}",
            f"hardware = {get_hardware_name(mem.hardware)}",
            f"reuse_id = {mem.reuse_id}"
        ]
        return f"#afir.mem<{', '.join(parts)}>"

    def gen_mem_queue_attr(self, que: MemQueueAttrDef) -> str:
        """Generate #afir.mem_queue attribute."""
        parts = [
            f"id = {que.id}",
            f"depth = {que.depth}",
            f"buf_num = {que.buf_num}"
        ]
        return f"#afir.mem_queue<{', '.join(parts)}>"

    def gen_mem_buf_attr(self, buf: MemBufAttrDef) -> str:
        """Generate #afir.mem_buf attribute."""
        return f"#afir.mem_buf<id = {buf.id}>"

    def gen_asc_tensor_attr(self, tensor: AscTensorDef) -> str:
        """Generate #afir.asc_tensor attribute."""
        if tensor.attr is None:
            return "#afir.asc_tensor<>"

        attr = tensor.attr
        parts = [
            f"dtype = {get_dtype_name(attr.dtype)}",
            f"axis_ids = {self.gen_array_i64(attr.axis_ids)}",
            f"repeats = {self.gen_array_str(attr.repeats)}",
            f"strides = {self.gen_array_str(attr.strides)}",
            f"vectorized_axis = {self.gen_array_i64(attr.vectorized_axis)}",
            f"vectorized_strides = {self.gen_array_str(attr.vectorized_strides)}"
        ]
        if attr.mem:
            parts.append(f"mem = {self.gen_mem_attr(attr.mem)}")
        if attr.que:
            parts.append(f"que = {self.gen_mem_queue_attr(attr.que)}")
        if attr.buf:
            parts.append(f"buf = {self.gen_mem_buf_attr(attr.buf)}")

        return f"#afir.asc_tensor<{', '.join(parts)}>"

    def gen_axis_attr(self, axis: AxisDef) -> str:
        """Generate #afir.axis attribute."""
        parts = [
            f"id = {axis.id}",
            f'name = "{escape_string(axis.name)}"',
            f"axis_type = {get_axis_type_name(axis.axis_type)}",
            f"bind_block = {str(axis.bind_block).lower()}",
            f'size = "{escape_string(axis.size)}"'
        ]
        if axis.align:
            parts.append(f'align = "{escape_string(axis.align)}"')
        parts.append(f"from = {self.gen_array_i64(axis.from_ids)}")
        parts.append(f"split_pair_other_id = {axis.split_pair_other_id}")
        return f"#afir.axis<{', '.join(parts)}>"

    def gen_sched_info_attr(self, sched: SchedInfoDef) -> str:
        """Generate #afir.sched attribute."""
        parts = [
            f"exec_order = {sched.exec_order}",
            f"axis = {self.gen_array_i64(sched.axis)}",
            f"loop_axis = {sched.loop_axis}",
            f"exec_condition = {get_exec_condition_name(sched.exec_condition)}"
        ]
        return f"#afir.sched<{', '.join(parts)}>"

    def gen_api_info_attr(self, api: ApiInfoDef) -> str:
        """Generate #afir.api attribute."""
        parts = [
            f"type = {get_api_type_name(api.type)}",
            f"compute_type = {get_compute_type_name(api.compute_type)}",
            f"unit = {get_compute_unit_name(api.unit)}"
        ]
        return f"#afir.api<{', '.join(parts)}>"

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

    def gen_asc_node_attr(self, node: AscNodeDef) -> str:
        """Generate #afir.asc_node attribute."""
        if node.attr is None:
            return "#afir.asc_node<>"

        attr = node.attr
        parts = [
            f'name = "{escape_string(attr.name)}"',
            f'type = "{escape_string(attr.type)}"'
        ]
        if attr.sched:
            parts.append(f"sched = {self.gen_sched_info_attr(attr.sched)}")
        if attr.api:
            parts.append(f"api = {self.gen_api_info_attr(attr.api)}")
        parts.append(f"ir_attr_def = {self.gen_ir_attr_dict(attr.ir_attr_def)}")
        parts.append("tmp_buffers = []")

        return f"#afir.asc_node<{', '.join(parts)}>"

    def gen_ir_def_attr(self, ir_def: Optional[IrDef]) -> str:
        """Generate #afir.ir_def attribute."""
        if ir_def is None:
            return "#afir.ir_def<>"

        parts = [
            f"input_names = {self.gen_array_str(ir_def.input_names)}",
            f"output_names = {self.gen_array_str(ir_def.output_names)}",
            f"input_ir_type = {self.gen_array_i64(ir_def.input_ir_type)}",
            f"output_ir_type = {self.gen_array_i64(ir_def.output_ir_type)}",
            f'type = "{escape_string(ir_def.type)}"',
            f"input_nums = {self.gen_array_i64(ir_def.input_nums)}",
            f"output_nums = {self.gen_array_i64(ir_def.output_nums)}"
        ]
        return f"#afir.ir_def<{', '.join(parts)}>"

    def gen_input_src_attr(self, input_src: AscInputSourceDef) -> str:
        """Generate #afir.input_src attribute."""
        return f'#afir.input_src<src_node_name = "{escape_string(input_src.src_node_name)}", src_out_index = {input_src.src_out_index}>'

    def gen_asc_graph_attr(self, graph_attr: AscGraphAttrGroupsDef) -> str:
        """Generate #afir.asc_graph attribute."""
        axis_parts = [self.gen_axis_attr(ax) for ax in graph_attr.axis]
        parts = [
            f"tiling_key = {graph_attr.tiling_key}",
            f"axis = [{', '.join(axis_parts)}]",
            f"type = {get_graph_type_name(graph_attr.type)}",
            f"size_var = {self.gen_array_str(graph_attr.size_var)}"
        ]
        return f"#afir.asc_graph<{', '.join(parts)}>"

    def gen_node_attr(self, node: AscNodeDef) -> str:
        """Generate complete node attribute dictionary."""
        input_src_list = [self.gen_input_src_attr(src) for src in node.input_src]
        output_list = [self.gen_asc_tensor_attr(out) for out in node.outputs]

        parts = [
            f"input_src = [{', '.join(input_src_list)}]",
            f"outputs = [{', '.join(output_list)}]",
            f"attr = {self.gen_asc_node_attr(node)}",
            f"ir_def = {self.gen_ir_def_attr(node.ir_def)}"
        ]
        return f"#afir.node<{', '.join(parts)}>"

    def get_mlir_type_from_dtype(self, dtype: int) -> str:
        """Get MLIR tensor type from dtype."""
        dtype_map = {
            DataType.DT_UNDEFINED: "tensor<*xf32>",
            DataType.DT_FLOAT: "tensor<*xf32>",
            DataType.DT_FLOAT16: "tensor<*xf16>",
            DataType.DT_INT8: "tensor<*xi8>",
            DataType.DT_UINT8: "tensor<*xui8>",
            DataType.DT_INT16: "tensor<*xi16>",
            DataType.DT_UINT16: "tensor<*xui16>",
            DataType.DT_INT32: "tensor<*xi32>",
            DataType.DT_INT64: "tensor<*xi64>",
            DataType.DT_UINT32: "tensor<*xui32>",
            DataType.DT_UINT64: "tensor<*xui64>",
            DataType.DT_BOOL: "tensor<*xi1>",
            DataType.DT_DOUBLE: "tensor<*xf64>",
            DataType.DT_BF16: "tensor<*xbf16>",
        }
        try:
            dt = DataType(dtype)
            return dtype_map.get(dt, "tensor<*xf32>")
        except ValueError:
            return "tensor<*xf32>"

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
        """Generate AFIR operation for a node."""
        if node.attr is None:
            return ""

        node_name = node.attr.name
        node_type = node.attr.type
        op_name = self.get_node_op_name(node_type)

        # Get result type from output tensor dtype
        result_type = "tensor<*xf32>"
        if node.outputs and node.outputs[0].attr:
            result_type = self.get_mlir_type_from_dtype(node.outputs[0].attr.dtype)

        # Get input SSA values
        input_values = []
        for src in node.input_src:
            src_name = src.src_node_name
            if src_name in self.node_map:
                input_values.append(f"%{self.node_map[src_name]}")
            else:
                input_values.append(f"%arg_{src_name}")

        # Generate result SSA value
        result_ssa = self.ssa_counter
        self.node_map[node_name] = result_ssa
        self.ssa_counter += 1

        # Build the operation
        indent = self.get_indent()
        node_attr = self.gen_node_attr(node)

        if len(input_values) == 0:
            # No inputs (e.g., Data node)
            return f"{indent}%{result_ssa} = \"{op_name}\"() {{{{\n{indent}  node_attr = {node_attr}\n{indent}}}}}: () -> {result_type}"
        elif len(input_values) == 1:
            # Single input
            return f"{indent}%{result_ssa} = \"{op_name}\"({input_values[0]}) {{{{\n{indent}  node_attr = {node_attr}\n{indent}}}}}: ({result_type}) -> {result_type}"
        else:
            # Multiple inputs
            inputs_str = ", ".join(input_values)
            types_str = ", ".join([result_type] * len(input_values))
            return f"{indent}%{result_ssa} = \"{op_name}\"({inputs_str}) {{{{\n{indent}  node_attr = {node_attr}\n{indent}}}}}: ({types_str}) -> {result_type}"

    def generate(self) -> str:
        """Generate complete AFIR MLIR text."""
        lines = []

        # Module header
        lines.append("// AFIR Dialect representation of AscGraph")
        lines.append(f'// Graph name: {self.graph.graph_name}')
        lines.append("")

        # Graph attributes as module attribute
        if self.graph.asc_graph_attr:
            graph_attr = self.gen_asc_graph_attr(self.graph.asc_graph_attr)
            lines.append(f"module attributes {{asc_graph_attr = {graph_attr}}} {{")
        else:
            lines.append("module {")

        self.indent = 1

        # Generate function with nodes
        lines.append(f'{self.get_indent()}func.func @{self.graph.graph_name or "main"}() {{')
        self.indent = 2

        # Generate operations for each node
        for node in self.graph.asc_node:
            op_str = self.gen_node_operation(node)
            if op_str:
                lines.append(op_str)

        # Return
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
    # Handle case where the entire content is escaped (e.g., {\"key\": \"value\"})
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
    # First, try to unescape if needed (handles double-escaped JSON)
    working_str = json_str.strip()

    # Check if the string starts with escaped braces
    if working_str.startswith('{\\'):
        working_str = unescape_json_string(working_str)

    # Try to parse as JSON
    try:
        data = json.loads(working_str)
    except json.JSONDecodeError:
        # If it fails, the input might already be protobuf text
        # Try unescaping one more time in case of nested escaping
        try:
            unescaped = unescape_json_string(working_str)
            data = json.loads(unescaped)
        except json.JSONDecodeError:
            return working_str

    # Navigate to find the ascgraph field
    # The structure is: compute_graph -> op (with type AscGraph) -> attr -> ascgraph -> value -> s
    if 'compute_graph' in data:
        compute_graph_str = data['compute_graph']
        # This is a protobuf text format string
        # Parse it to find the ascgraph attribute
        parser = ProtobufTextParser(compute_graph_str)
        cg_data = parser.parse()

        # Look through ops for AscGraph type
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

    # If we can't find it in the expected structure, return as-is
    return working_str


def convert_ascgraph_to_afir(input_text: str) -> str:
    """Convert AscGraph JSON/text to AFIR MLIR text."""
    # Extract the ascgraph protobuf text from JSON if needed
    ascgraph_text = extract_ascgraph_from_json(input_text)

    # Parse the protobuf text format
    parser = ProtobufTextParser(ascgraph_text)
    data = parser.parse()

    # Parse into AscGraphDef
    graph = parse_asc_graph(data)

    # Generate AFIR MLIR text
    generator = AFIRGenerator(graph)
    return generator.generate()


def main():
    parser = argparse.ArgumentParser(
        description='Convert AscGraph JSON to AFIR MLIR text representation'
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
        sys.exit(1)

    # Write output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
    else:
        print(output_text)


if __name__ == '__main__':
    main()
