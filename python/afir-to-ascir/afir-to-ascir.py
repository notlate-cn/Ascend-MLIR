#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., 2026 Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
"""
AFIR Dialect MLIR Text to AscGraph Converter

This script converts AFIR dialect MLIR text representation to AscGraph
protobuf text format.

This is the reverse conversion of ascir_to_afir.py.
"""

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
    """DataType enum matching proto definition."""
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
    DT_BF16 = 27


class Position(IntEnum):
    """Position enum (fused with AllocType)."""
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
    """Graph type enum."""
    COMPUTE = 0
    Invalid = 1


# ============================================================================
# Data Classes for AscGraph structures
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
    depth: int = -1
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
    allow_unaligned_tail: bool = True
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
# AFIR MLIR Parser
# ============================================================================

class AFIRMLIRParser:
    """Parser for AFIR MLIR text format."""

    def __init__(self, text: str, api_config: Optional[Dict[str, tuple]] = None):
        self.text = text
        self.lines = text.split('\n')
        self.graph = AscGraphDef()
        self.ssa_map: Dict[str, AscNodeDef] = {}  # map SSA value to node
        self.arg_info: Dict[str, tuple] = {}  # map arg name to (Data node, dtype, shape)
        self.data_nodes: List[AscNodeDef] = []
        self.compute_nodes: List[AscNodeDef] = []
        self.all_nodes: List[AscNodeDef] = []  # All nodes in order
        self.exec_order_counter = 1

        # Default API configuration for operation types
        # Format: {op_name: (api_type, compute_type, unit)}
        default_config = {
            'load': (1, 11, 2),
            'store': (1, 1, 2),
            'add': (1, 3, 5),
            'sub': (1, 3, 5),
            'mul': (1, 3, 5),
            'div': (1, 3, 5),
            'broadcast': (2, 11, 7),
            'exp': (1, 3, 5),
            'log': (1, 3, 5),
            'sqrt': (1, 3, 5),
            'relu': (1, 3, 5),
            'sigmoid': (1, 3, 5),
            'tanh': (1, 3, 5),
        }

        # Merge user config with defaults (user config takes precedence)
        self.api_config = default_config.copy()
        if api_config:
            self.api_config.update(api_config)

    def parse_mlir_type(self, type_str: str) -> tuple[int, List[int]]:
        """Parse MLIR tensor type to dtype and shape."""
        # tensor<20x31xf32> -> (DT_FLOAT, [20, 31])
        match = re.search(r'tensor<([^>]+)>', type_str)
        if not match:
            return (DataType.DT_FLOAT, [])

        content = match.group(1)
        parts = content.split('x')

        # Get element type from last part
        elem_type = parts[-1]
        dtype_map = {
            'f32': DataType.DT_FLOAT,
            'f16': DataType.DT_FLOAT16,
            'bf16': DataType.DT_BF16,
            'i8': DataType.DT_INT8,
            'ui8': DataType.DT_UINT8,
            'i16': DataType.DT_INT16,
            'ui16': DataType.DT_UINT16,
            'i32': DataType.DT_INT32,
            'i64': DataType.DT_INT64,
            'ui32': DataType.DT_UINT32,
            'ui64': DataType.DT_UINT64,
            'i1': DataType.DT_BOOL,
            'f64': DataType.DT_DOUBLE,
        }
        dtype = dtype_map.get(elem_type, DataType.DT_FLOAT)

        # Get shape from all parts except last
        shape = []
        for part in parts[:-1]:
            if part == '?' or part == '*':
                shape.append(-1)
            else:
                try:
                    shape.append(int(part))
                except ValueError:
                    shape.append(-1)

        return (dtype, shape)

    def parse_position(self, pos_str: str) -> tuple[int, int]:
        """Parse position string to position and alloc_type."""
        # <gm> -> (0, 0), <l1> -> (0, 1), <l2> -> (0, 2)
        pos_str = pos_str.strip('<>')
        pos_map = {
            'gm': (Position.GM, 0),
            'vector_in': (Position.VECTOR_IN, 0),
            'vector_out': (Position.VECTOR_OUT, 0),
            'vector_calc': (Position.VECTOR_CALC, 0),
            'l1': (Position.GM, 1),
            'l2': (Position.GM, 2),
            'l0a': (Position.L0A, 0),
            'l0b': (Position.L0B, 0),
            'l0c': (Position.L0C, 0),
        }
        return pos_map.get(pos_str.lower(), (Position.GM, 0))

    def parse_asc_tensor_attr(self, attr_str: str) -> AscTensorAttrGroupsDef:
        """Parse #afir.asc_tensor<...> attribute."""
        result = AscTensorAttrGroupsDef()
        result.mem = MemAttrDef()
        result.que = MemQueueAttrDef()
        result.buf = MemBufAttrDef()
        result.opt = MemOptAttrDef()

        # Extract content between < and >
        match = re.search(r'#afir\.asc_tensor<([^>]+)>', attr_str)
        if not match:
            return result

        content = match.group(1)

        # Parse tensor_id
        tensor_id_match = re.search(r'tensor_id\s*=\s*(-?\d+)', content)
        if tensor_id_match:
            result.mem.tensor_id = int(tensor_id_match.group(1))

        # Parse reuse_id
        reuse_id_match = re.search(r'reuse_id\s*=\s*(-?\d+)', content)
        if reuse_id_match:
            result.mem.reuse_id = int(reuse_id_match.group(1))

        # Parse position
        position_match = re.search(r'position\s*=\s*<([^>]+)>', content)
        if position_match:
            position, alloc_type = self.parse_position(f"<{position_match.group(1)}>")
            result.mem.position = position
            result.mem.alloc_type = alloc_type

        # Parse position_id
        position_id_match = re.search(r'position_id\s*=\s*(-?\d+)', content)
        if position_id_match:
            pid = int(position_id_match.group(1))
            # Assign to que or buf depending on context
            result.que.id = pid
            result.buf.id = pid

        return result

    def parse_axis_attr(self, attr_str: str) -> AxisDef:
        """Parse <id=0, name=\"z0\", ...> axis attribute."""
        axis = AxisDef()

        # Parse id
        id_match = re.search(r'id\s*=\s*(\d+)', attr_str)
        if id_match:
            axis.id = int(id_match.group(1))

        # Parse name
        name_match = re.search(r'name\s*=\s*"([^"]+)"', attr_str)
        if name_match:
            axis.name = name_match.group(1)

        # Parse axis_type
        type_match = re.search(r'axis_type\s*=\s*(\w+)', attr_str)
        if type_match:
            type_name = type_match.group(1)
            try:
                axis.axis_type = AxisType[type_name].value
            except KeyError:
                axis.axis_type = AxisType.Original.value

        # Parse bind_block
        if 'bind_block' in attr_str:
            bind_match = re.search(r'bind_block\s*=\s*(true|false)', attr_str)
            if bind_match:
                axis.bind_block = bind_match.group(1) == 'true'

        # Parse size
        size_match = re.search(r'size\s*=\s*"([^"]+)"', attr_str)
        if size_match:
            axis.size = size_match.group(1)

        # Parse align
        align_match = re.search(r'align\s*=\s*"([^"]+)"', attr_str)
        if align_match:
            axis.align = align_match.group(1)

        # Parse from
        from_match = re.search(r'from\s*=\s*\[([^\]]+)\]', attr_str)
        if from_match:
            axis.from_ids = [int(x.strip()) for x in from_match.group(1).split(',')]

        return axis

    def parse_graph_attr(self, attr_str: str) -> AscGraphAttrGroupsDef:
        """Parse #afir.asc_graph<...> attribute."""
        result = AscGraphAttrGroupsDef()

        # Extract content between < and >
        match = re.search(r'#afir\.asc_graph<(.+)>', attr_str, re.DOTALL)
        if not match:
            return result

        content = match.group(1)

        # Parse axes (array of axis attributes)
        axes_match = re.search(r'axes\s*=\s*\[([^\]]+(?:\][^\]]*)*)\]', content)
        if axes_match:
            axes_content = axes_match.group(1)
            # Find all <...> axis definitions
            axis_pattern = r'<[^>]+>'
            for axis_match in re.finditer(axis_pattern, axes_content):
                axis = self.parse_axis_attr(axis_match.group(0))
                result.axis.append(axis)

        # Parse type
        type_match = re.search(r'type\s*=\s*(\w+)', content)
        if type_match:
            type_name = type_match.group(1)
            try:
                result.type = AscGraphType[type_name].value
            except KeyError:
                result.type = AscGraphType.COMPUTE.value

        return result

    def parse_ir_attr_def(self, attr_str: str) -> AscIrAttrDef:
        """Parse ir_attr_def = {...} dictionary."""
        result = AscIrAttrDef()

        # Extract content between { and }
        match = re.search(r'\{([^}]+)\}', attr_str)
        if not match:
            return result

        content = match.group(1)

        # Parse key-value pairs - support both quoted and unquoted keys
        # Patterns: offset = "0" or "offset" = "0" or "offset" = 123
        kv_pattern = r'(?:"([^"]+)"|(\w+))\s*=\s*(?:"([^"]+)"|(\d+)|(-?\d+\.\d+)|(true|false))'
        for match in re.finditer(kv_pattern, content):
            key = match.group(1) or match.group(2)  # Quoted or unquoted key
            str_val = match.group(3)
            int_val = match.group(4)
            float_val = match.group(5)
            bool_val = match.group(6)

            if str_val is not None:
                # Could be string or expression
                result.attr[key] = IrAttrValue(expression=str_val)
            elif int_val is not None:
                result.attr[key] = IrAttrValue(i=int(int_val))
            elif float_val is not None:
                result.attr[key] = IrAttrValue(f=float(float_val))
            elif bool_val is not None:
                result.attr[key] = IrAttrValue(b=(bool_val == 'true'))

        return result

    def parse_func_args(self, func_line: str) -> tuple[List[AscNodeDef], Dict[str, tuple]]:
        """Parse function arguments to create Data nodes."""
        data_nodes = []
        arg_info = {}  # Map arg name to (Data node, type info)

        # Extract function signature: func.func @name(%arg0: type, %arg1: type) -> type
        match = re.search(r'@(\w+)\(([^)]*)\)', func_line)
        if not match:
            return data_nodes, arg_info

        func_name = match.group(1)
        args_str = match.group(2)

        if not args_str.strip():
            return data_nodes, arg_info

        # Parse each argument
        arg_pattern = r'%arg(\d+):\s*([^,]+)'
        for arg_match in re.finditer(arg_pattern, args_str):
            arg_idx = int(arg_match.group(1))
            arg_type = arg_match.group(2).strip()

            dtype, shape = self.parse_mlir_type(arg_type)

            # Create Data node
            node = AscNodeDef()
            node.attr = AscNodeAttrGroupsDef()
            node.attr.name = f"{func_name}/Data_{arg_idx}"  # Will be renumbered later
            node.attr.type = "Data"
            node.attr.sched = SchedInfoDef()
            node.attr.api = ApiInfoDef()
            node.attr.api.compute_type = 11  # Data nodes have compute_type = 11
            node.attr.ir_attr_def = AscIrAttrDef()
            node.attr.ir_attr_def.attr["index"] = IrAttrValue(i=arg_idx)

            # Data nodes after the first have exec_order (will be renumbered later)
            if arg_idx > 0:
                node.attr.sched.exec_order = 999  # Temporary, will be renumbered

            # Create output tensor
            tensor_attr = AscTensorAttrGroupsDef()
            tensor_attr.dtype = dtype

            # Create axis_ids, repeats, strides based on shape
            if shape:
                for i, dim in enumerate(shape):
                    tensor_attr.axis_ids.append(i)
                    tensor_attr.repeats.append(str(dim) if dim > 0 else "?")

                # Calculate strides (row-major), with special handling for broadcast-like shapes
                if len(shape) > 0:
                    # For shapes like (1, 31), strides should be (0, 1) to enable broadcast
                    for i in range(len(shape)):
                        if shape[i] == 1:
                            tensor_attr.strides.append("0")
                        else:
                            # Calculate stride for this dimension
                            stride = 1
                            for j in range(i + 1, len(shape)):
                                if shape[j] > 0 and shape[j] != 1:
                                    stride *= shape[j]
                            tensor_attr.strides.append(str(stride))

            tensor_attr.mem = MemAttrDef()
            tensor_attr.que = MemQueueAttrDef()
            tensor_attr.buf = MemBufAttrDef()
            tensor_attr.opt = MemOptAttrDef()

            node.outputs = [AscTensorDef(attr=tensor_attr)]

            # Create IrDef
            node.ir_def = IrDef()
            node.ir_def.output_names = ["y"]
            node.ir_def.output_ir_type = [0]
            node.ir_def.type = "Data"
            node.ir_def.output_nums = [1]

            data_nodes.append(node)
            arg_info[f"arg{arg_idx}"] = (node, dtype, shape)

        return data_nodes, arg_info

    def _parse_api_attributes(self, node: AscNodeDef, op_name: str, attrs: str):
        """
        Parse API attributes from AFIR operation attributes.
        This is a generic parser that works for any operation type.

        The API configuration can be customized by passing api_config to the parser.
        For unknown operations, generic defaults are used.
        """
        # Apply values from configuration if operation is known
        if op_name in self.api_config:
            api_type, compute_type, unit = self.api_config[op_name]
            node.attr.api.type = api_type
            node.attr.api.compute_type = compute_type
            node.attr.api.unit = unit

            # Special case: Broadcast operations don't have exec_order
            if op_name == 'broadcast':
                node.attr.sched.exec_order = -1
        else:
            # For unknown operations, use generic defaults
            # This allows the script to handle any AFIR operation
            node.attr.api.type = 1
            node.attr.api.compute_type = 11
            node.attr.api.unit = 0

            # Log unknown operation for debugging
            import sys
            print(f"Warning: Unknown operation '{op_name}', using default API config", file=sys.stderr)

    def _parse_ir_attr_def_from_afir(self, node: AscNodeDef, op_name: str, attrs: str):
        """
        Parse ir_attr_def from AFIR attributes.
        This is generic and works for any operation that has ir_attr_def in AFIR.
        """
        ir_attr_match = re.search(r'ir_attr_def\s*=\s*\{([^}]*)\}', attrs)
        if ir_attr_match:
            ir_content = ir_attr_match.group(1).strip()
            if ir_content:
                node.attr.ir_attr_def = self.parse_ir_attr_def(f"{{{ir_content}}}")
            else:
                # Empty ir_attr_def
                node.attr.ir_attr_def = AscIrAttrDef()
        else:
            # Some operations may need ir_attr_def even if not present in AFIR
            # Handle known cases, but this is operation-specific
            if op_name == 'store':
                node.attr.ir_attr_def = AscIrAttrDef()
            elif op_name == 'load':
                # Load operations typically have an offset attribute
                node.attr.ir_attr_def = AscIrAttrDef()
                node.attr.ir_attr_def.attr["offset"] = IrAttrValue(expression="0")

    def parse_operation(self, line: str) -> Optional[AscNodeDef]:
        """Parse an AFIR operation line."""
        # %1 = afir.load %arg0 {...} : tensor<...> -> tensor<...>
        # Need to match nested braces for ir_attr_def
        match = re.match(r'\s*%(\w+)\s*=\s*afir\.(\w+)\s+([^{]*)\s*\{(.+)\}\s*:', line)
        if not match:
            return None

        ssa_val = match.group(1)
        op_name = match.group(2)
        operands = match.group(3).strip()
        attrs = match.group(4)

        # Get input and return types
        type_match = re.search(r':\s*([^-]+)\s*->\s*(.+)', line)
        if type_match:
            input_types_str = type_match.group(1).strip()
            result_type = type_match.group(2).strip()
        else:
            input_types_str = ""
            result_type = "tensor<*xf32>"

        dtype, shape = self.parse_mlir_type(result_type)

        # Create node
        node = AscNodeDef()
        node.attr = AscNodeAttrGroupsDef()
        node.attr.name = f"{self.graph.graph_name}/{op_name.capitalize()}_{self.exec_order_counter}"
        node.attr.type = op_name.capitalize()
        node.attr.sched = SchedInfoDef()
        node.attr.sched.exec_order = self.exec_order_counter
        node.attr.api = ApiInfoDef()

        # Parse API attributes from AFIR if available, otherwise use defaults
        # This makes the conversion generic for any operation type
        self._parse_api_attributes(node, op_name, attrs)

        # Parse ir_attr_def from AFIR attributes
        self._parse_ir_attr_def_from_afir(node, op_name, attrs)

        # Parse outputs attribute
        outputs_match = re.search(r'outputs\s*=\s*\[([^\]]+)\]', attrs)
        tensor_attr = AscTensorAttrGroupsDef()
        if outputs_match:
            tensor_attr = self.parse_asc_tensor_attr(f"#afir.asc_tensor<{outputs_match.group(1)}>")
        else:
            tensor_attr.mem = MemAttrDef()
            tensor_attr.que = MemQueueAttrDef()
            tensor_attr.buf = MemBufAttrDef()
            tensor_attr.opt = MemOptAttrDef()

        tensor_attr.dtype = dtype

        # Set axis_ids, repeats, strides based on shape
        if shape:
            for i, dim in enumerate(shape):
                tensor_attr.axis_ids.append(i)
                tensor_attr.repeats.append(str(dim) if dim > 0 else "?")

            # Calculate strides (with special handling for broadcast-like shapes)
            if len(shape) > 0:
                for i in range(len(shape)):
                    if shape[i] == 1:
                        tensor_attr.strides.append("0")
                    else:
                        stride = 1
                        for j in range(i + 1, len(shape)):
                            if shape[j] > 0 and shape[j] != 1:
                                stride *= shape[j]
                        tensor_attr.strides.append(str(stride))

        node.outputs = [AscTensorDef(attr=tensor_attr)]

        # Parse input operands
        if operands:
            operand_list = [op.strip() for op in operands.split(',')]
            for operand in operand_list:
                if operand.startswith('%'):
                    src_name = operand[1:]  # Remove %

                    # Check if this is an argument - map to Data node
                    if src_name in self.arg_info:
                        data_node, arg_dtype, arg_shape = self.arg_info[src_name]
                        # For Load operations, input comes from Data node
                        node.input_src.append(AscInputSourceDef(
                            src_node_name=data_node.attr.name,
                            src_out_index=0
                        ))
                    else:
                        src_node = self.ssa_map.get(src_name)
                        if src_node:
                            node.input_src.append(AscInputSourceDef(
                                src_node_name=src_node.attr.name,
                                src_out_index=0
                            ))

        # Create IrDef
        node.ir_def = IrDef()
        num_inputs = len(node.input_src)
        if num_inputs == 0:
            node.ir_def.output_names = ["y"]
        elif num_inputs == 1:
            node.ir_def.input_names = ["x"]
            node.ir_def.output_names = ["y"]
            node.ir_def.input_ir_type = [0]
            node.ir_def.input_nums = [1]
        else:
            node.ir_def.input_names = [f"x{i+1}" for i in range(num_inputs)]
            node.ir_def.output_names = ["y"]
            node.ir_def.input_ir_type = [0] * num_inputs
            node.ir_def.input_nums = [1] * num_inputs

        node.ir_def.output_ir_type = [0]
        node.ir_def.type = node.attr.type
        node.ir_def.output_nums = [1]

        # Store in SSA map
        self.ssa_map[ssa_val] = node
        self.exec_order_counter += 1

        return node

    def parse(self) -> AscGraphDef:
        """Parse complete AFIR MLIR text."""
        in_function = False

        for line in self.lines:
            line = line.strip()

            # Parse module attributes for graph attributes
            if 'afir.asc_graph_attr' in line:
                # Extract the entire attribute (may span multiple lines)
                attr_start = line.find('#afir.asc_graph<')
                if attr_start != -1:
                    # Find matching closing >
                    depth = 0
                    attr_end = -1
                    for i in range(attr_start, len(line)):
                        if line[i] == '<':
                            depth += 1
                        elif line[i] == '>':
                            depth -= 1
                            if depth == 0:
                                attr_end = i + 1
                                break

                    if attr_end != -1:
                        attr_str = line[attr_start:attr_end]
                        self.graph.asc_graph_attr = self.parse_graph_attr(attr_str)

            # Parse function signature
            if 'func.func @' in line:
                # Extract function name
                func_match = re.search(r'@(\w+)', line)
                if func_match:
                    self.graph.graph_name = func_match.group(1)

                # Parse function arguments to create Data nodes
                self.data_nodes, self.arg_info = self.parse_func_args(line)
                in_function = True
                continue

            # Parse operations inside function
            if in_function and line.startswith('%'):
                node = self.parse_operation(line)
                if node:
                    self.all_nodes.append(node)

            # Parse return statement
            if 'return' in line and in_function:
                # Create Output node
                return_match = re.search(r'return\s+%(\w+)', line)
                if return_match:
                    ret_val = return_match.group(1)
                    src_node = self.ssa_map.get(ret_val)

                    if src_node:
                        output_node = AscNodeDef()
                        output_node.attr = AscNodeAttrGroupsDef()
                        output_node.attr.name = f"{self.graph.graph_name}/Output_{self.exec_order_counter}"
                        output_node.attr.type = "Output"
                        output_node.attr.sched = SchedInfoDef()
                        output_node.attr.sched.exec_order = self.exec_order_counter
                        output_node.attr.api = ApiInfoDef()
                        output_node.attr.api.compute_type = 11  # Output节点需要compute_type
                        output_node.attr.ir_attr_def = AscIrAttrDef()
                        output_node.attr.ir_attr_def.attr["index"] = IrAttrValue(i=0)

                        output_node.input_src.append(AscInputSourceDef(
                            src_node_name=src_node.attr.name,
                            src_out_index=0
                        ))

                        # Create output tensor with empty attributes
                        tensor_attr = AscTensorAttrGroupsDef()
                        tensor_attr.mem = MemAttrDef()
                        tensor_attr.que = MemQueueAttrDef()
                        tensor_attr.buf = MemBufAttrDef()
                        tensor_attr.opt = MemOptAttrDef()
                        output_node.outputs = [AscTensorDef(attr=tensor_attr)]

                        # Create IrDef
                        output_node.ir_def = IrDef()
                        output_node.ir_def.input_names = ["x"]
                        output_node.ir_def.output_names = ["y"]
                        output_node.ir_def.input_ir_type = [0]
                        output_node.ir_def.output_ir_type = [0]
                        output_node.ir_def.type = "Output"
                        output_node.ir_def.input_nums = [1]
                        output_node.ir_def.output_nums = [1]

                        self.all_nodes.append(output_node)

                in_function = False

        # Combine all nodes - interleave Data and Load nodes, then add compute nodes
        final_nodes = []
        node_counter = 0
        exec_order_counter = 1  # Start from 1
        name_map = {}  # Track old name -> new name for reference updates

        # Add Data and Load nodes alternately
        for data_node in self.data_nodes:
            # Track old name
            old_data_name = data_node.attr.name

            # Renumber Data node
            new_data_name = f"{self.graph.graph_name}/Data_{node_counter}"
            data_node.attr.name = new_data_name
            name_map[old_data_name] = new_data_name

            if node_counter > 0:
                data_node.attr.sched.exec_order = exec_order_counter
                exec_order_counter += 1

            final_nodes.append(data_node)
            node_counter += 1

            # Find corresponding Load node (looks for Load ops that reference this Data)
            for node in self.all_nodes:
                if (node.attr and node.attr.type == "Load" and
                    node.input_src and len(node.input_src) > 0):
                    # Check if this Load references the Data node (by old name)
                    if node.input_src[0].src_node_name == old_data_name:
                        # Track old Load name
                        old_load_name = node.attr.name

                        # Renumber Load node
                        new_load_name = f"{self.graph.graph_name}/Load_{node_counter}"
                        node.attr.name = new_load_name
                        name_map[old_load_name] = new_load_name

                        # Update Load's input to use new Data name
                        node.input_src[0].src_node_name = new_data_name

                        node.attr.sched.exec_order = exec_order_counter
                        exec_order_counter += 1

                        final_nodes.append(node)
                        node_counter += 1
                        break

        # Add remaining compute nodes
        for node in self.all_nodes:
            if node.attr and node.attr.type not in ["Data", "Load"]:
                # Track old name
                old_name = node.attr.name
                op_type = node.attr.type

                # Renumber compute node
                new_name = f"{self.graph.graph_name}/{op_type}_{node_counter}"
                node.attr.name = new_name
                name_map[old_name] = new_name

                # Update this node's input references
                for inp in node.input_src:
                    if inp.src_node_name in name_map:
                        inp.src_node_name = name_map[inp.src_node_name]

                # Update exec_order
                if node.attr.sched.exec_order == -1:
                    # Broadcast keeps -1, don't increment exec_order_counter
                    pass
                else:
                    node.attr.sched.exec_order = exec_order_counter
                    exec_order_counter += 1

                final_nodes.append(node)
                node_counter += 1

        self.graph.asc_node = final_nodes

        # Set axis for sched based on graph axes
        if self.graph.asc_graph_attr:
            axis_ids = [ax.id for ax in self.graph.asc_graph_attr.axis]
            for node in self.graph.asc_node:
                if node.attr and node.attr.sched:
                    node.attr.sched.axis = axis_ids

        return self.graph


# ============================================================================
# AscGraph Protobuf Text Generator
# ============================================================================

def generate_proto_text(graph: AscGraphDef) -> str:
    """Generate protobuf text format from AscGraphDef."""
    lines = []

    # Generate graph attributes
    if graph.asc_graph_attr:
        lines.append("asc_graph_attr {")
        lines.append(f"  tiling_key: {graph.asc_graph_attr.tiling_key}")

        for axis in graph.asc_graph_attr.axis:
            lines.append("  axis {")
            if axis.id != 0:
                lines.append(f"    id: {axis.id}")
            if axis.name:
                lines.append(f"    name: \"{axis.name}\"")
            if axis.size:
                lines.append(f"    size: \"{axis.size}\"")
            if axis.align:
                lines.append(f"    align: \"{axis.align}\"")
            lines.append(f"    allow_unaligned_tail: {str(axis.allow_unaligned_tail).lower()}")
            lines.append("  }")

        lines.append("}")

    # Generate nodes
    for node in graph.asc_node:
        lines.append("asc_node {")

        # Input sources
        for src in node.input_src:
            lines.append("  input_src {")
            lines.append(f"    src_node_name: \"{src.src_node_name}\"")
            if src.src_out_index != 0:
                lines.append(f"    src_out_index: {src.src_out_index}")
            lines.append("  }")

        # Outputs
        for output in node.outputs:
            if output.attr:
                lines.append("  outputs {")
                lines.append("    attr {")

                attr = output.attr
                for axis_id in attr.axis_ids:
                    lines.append(f"      axis_ids: {axis_id}")

                for repeat in attr.repeats:
                    lines.append(f"      repeats: \"{repeat}\"")

                for stride in attr.strides:
                    lines.append(f"      strides: \"{stride}\"")

                if attr.mem:
                    lines.append("      mem {")
                    lines.append(f"        tensor_id: {attr.mem.tensor_id}")
                    if attr.mem.alloc_type != 0:
                        lines.append(f"        alloc_type: {attr.mem.alloc_type}")
                    if attr.mem.position != 0:
                        lines.append(f"        position: {attr.mem.position}")
                    if attr.mem.hardware != 0:
                        lines.append(f"        hardware: {attr.mem.hardware}")
                    if attr.mem.reuse_id != -1:
                        lines.append(f"        reuse_id: {attr.mem.reuse_id}")
                    lines.append("      }")

                if attr.que:
                    lines.append("      que {")
                    lines.append(f"        id: {attr.que.id}")
                    lines.append(f"        depth: {attr.que.depth}")
                    lines.append(f"        buf_num: {attr.que.buf_num}")
                    lines.append("      }")

                if attr.buf:
                    lines.append("      buf {")
                    lines.append(f"        id: {attr.buf.id}")
                    lines.append("      }")

                if attr.opt:
                    lines.append("      opt {")
                    lines.append(f"        reuse_id: {attr.opt.reuse_id}")
                    lines.append(f"        ref_tensor: {attr.opt.ref_tensor}")
                    lines.append(f"        merge_scope: {attr.opt.merge_scope}")
                    lines.append("      }")

                lines.append("    }")
                lines.append("  }")

        # Attributes
        if node.attr:
            lines.append("  attr {")
            lines.append(f"    name: \"{node.attr.name}\"")
            lines.append(f"    type: \"{node.attr.type}\"")

            if node.attr.sched:
                lines.append("    sched {")
                # Output exec_order for all nodes, even if -1 (for Broadcast)
                if node.attr.sched.exec_order != -1 or node.attr.type == "Broadcast":
                    if node.attr.sched.exec_order != -1 and node.attr.type != "Broadcast":
                        lines.append(f"      exec_order: {node.attr.sched.exec_order}")
                    elif node.attr.type == "Broadcast" and node.attr.sched.exec_order == -1:
                        lines.append(f"      exec_order: -1")
                    # For first Data node, don't output exec_order
                    elif "Data_0" not in node.attr.name:
                        lines.append(f"      exec_order: {node.attr.sched.exec_order}")
                for axis_id in node.attr.sched.axis:
                    lines.append(f"      axis: {axis_id}")
                lines.append(f"      loop_axis: {node.attr.sched.loop_axis}")
                if node.attr.sched.exec_condition != 0:
                    lines.append(f"      exec_condition: {node.attr.sched.exec_condition}")
                lines.append("    }")

            if node.attr.api:
                lines.append("    api {")
                if node.attr.api.type != 0:
                    lines.append(f"      type: {node.attr.api.type}")
                # Output compute_type for Data, Broadcast, and Output nodes
                if node.attr.type == "Data":
                    lines.append(f"      compute_type: 11")
                elif node.attr.type == "Broadcast":
                    lines.append(f"      compute_type: 11")
                elif node.attr.type == "Output":
                    lines.append(f"      compute_type: 11")
                elif node.attr.api.compute_type != 11:
                    lines.append(f"      compute_type: {node.attr.api.compute_type}")
                if node.attr.api.unit != 0:
                    lines.append(f"      unit: {node.attr.api.unit}")
                lines.append("    }")

            if node.attr.ir_attr_def is not None:
                if node.attr.ir_attr_def.attr:
                    lines.append("    ir_attr_def {")
                    for key, value in node.attr.ir_attr_def.attr.items():
                        lines.append("      attr {")
                        lines.append(f"        key: \"{key}\"")
                        lines.append("        value {")
                        if value.i is not None:
                            lines.append(f"          i: {value.i}")
                        elif value.s is not None:
                            lines.append(f"          s: \"{value.s}\"")
                        elif value.expression is not None:
                            lines.append(f"          expression: \"{value.expression}\"")
                        elif value.b is not None:
                            lines.append(f"          b: {str(value.b).lower()}")
                        elif value.f is not None:
                            lines.append(f"          f: {value.f}")
                        lines.append("        }")
                        lines.append("      }")
                    lines.append("    }")
                elif node.attr.type == "Store":
                    # Store operations have empty ir_attr_def
                    lines.append("    ir_attr_def {")
                    lines.append("    }")

            lines.append("  }")

        # IrDef
        if node.ir_def:
            lines.append("  ir_def {")
            for name in node.ir_def.input_names:
                lines.append(f"    input_names: \"{name}\"")
            for name in node.ir_def.output_names:
                lines.append(f"    output_names: \"{name}\"")
            for ir_type in node.ir_def.input_ir_type:
                lines.append(f"    input_ir_type: {ir_type}")
            for ir_type in node.ir_def.output_ir_type:
                lines.append(f"    output_ir_type: {ir_type}")
            if node.ir_def.type:
                lines.append(f"    type: \"{node.ir_def.type}\"")
            for num in node.ir_def.input_nums:
                lines.append(f"    input_nums: {num}")
            for num in node.ir_def.output_nums:
                lines.append(f"    output_nums: {num}")
            lines.append("  }")

        lines.append("}")

    # Graph name
    if graph.graph_name:
        lines.append(f"graph_name: \"{graph.graph_name}\"")

    return "\n".join(lines)


# ============================================================================
# Main Entry Point
# ============================================================================

def convert_afir_to_ascgraph(input_text: str, api_config: Optional[Dict[str, tuple]] = None) -> str:
    """
    Convert AFIR MLIR text to AscGraph protobuf text format.

    Args:
        input_text: AFIR MLIR text to convert
        api_config: Optional API configuration for operation types.
                   Format: {op_name: (api_type, compute_type, unit)}
                   If not provided, uses default configuration.

    Returns:
        AscGraph protobuf text format
    """
    parser = AFIRMLIRParser(input_text, api_config)
    graph = parser.parse()
    return generate_proto_text(graph)


def main():
    argparser = argparse.ArgumentParser(
        description='Convert AFIR MLIR text to AscGraph protobuf text format',
        epilog='''
Examples:
  # Basic usage
  python afir-to-ascir.py input.afir -o output.txt

  # With custom API config
  python afir-to-ascir.py input.afir -o output.txt --api-config config.json

API config format (JSON):
  {
    "op_name": [api_type, compute_type, unit],
    "custom_op": [1, 3, 5]
  }
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        'input',
        nargs='?',
        help='Input file path (reads from stdin if not provided)'
    )
    argparser.add_argument(
        '-o', '--output',
        help='Output file path (writes to stdout if not provided)'
    )
    argparser.add_argument(
        '--api-config',
        help='Path to JSON file with API configuration for custom operations'
    )

    args = argparser.parse_args()

    # Read input
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_text = f.read()
    else:
        input_text = sys.stdin.read()

    # Load API config if provided
    api_config = None
    if args.api_config:
        try:
            with open(args.api_config, 'r', encoding='utf-8') as f:
                import json
                config_data = json.load(f)
                # Convert lists to tuples
                api_config = {k: tuple(v) for k, v in config_data.items()}
        except Exception as e:
            print(f"Error loading API config: {e}", file=sys.stderr)
            sys.exit(1)

    # Convert
    try:
        output_text = convert_afir_to_ascgraph(input_text, api_config)
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
