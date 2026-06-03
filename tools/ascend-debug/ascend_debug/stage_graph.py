from __future__ import annotations

import html
import os
import pathlib
import re
import shutil
import subprocess
from typing import Any

from ascend_debug import layout


SSA_VALUE_RE = re.compile(r"%[A-Za-z0-9_.$-]+")
FUNC_START_RE = re.compile(r"func\.func\s+@(?P<name>[A-Za-z0-9_.$-]+)\s*\(")
OP_RE = re.compile(
    r"^\s*(?P<results>%[A-Za-z0-9_.$-]+(?:\s*,\s*%[A-Za-z0-9_.$-]+)*)\s*=\s*(?P<op>[A-Za-z_][A-Za-z0-9_.]*)"
)
RESULTLESS_OP_RE = re.compile(r"^\s*(?P<op>[A-Za-z_][A-Za-z0-9_.]*)\b")
RETURN_RE = re.compile(r"^\s*return\b")
BODY_OP_RE = re.compile(
    r"^\s*(?:%[A-Za-z0-9_.$-]+(?:\s*,\s*%[A-Za-z0-9_.$-]+)*\s*=\s*)?"
    r"(?P<op>[A-Za-z_][A-Za-z0-9_.]*)\b"
)


def _cell(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _rel_no_ext(rel_path: str) -> str:
    path = pathlib.PurePosixPath(rel_path)
    return f"{path.stem}"


def _stage_graph_rel_paths(stage_rel_path: str) -> tuple[str, str]:
    stem = _rel_no_ext(stage_rel_path)
    return (
        f"graphs/stages/{stem}.graph.json",
        f"views/graphs/stages/{stem}.graph.html",
    )


def _find_mlir_stage_graph_tool() -> pathlib.Path | None:
    env_tool = os.environ.get("ASCEND_STAGE_GRAPH_TOOL")
    use_default_tool = os.environ.get("ASCEND_DEBUG_USE_MLIR_STAGE_GRAPH", "").lower()
    candidates = []
    if env_tool:
        candidates.append(pathlib.Path(env_tool))
    elif use_default_tool not in {"1", "true", "yes", "on"}:
        return None
    package_dir = pathlib.Path(__file__).resolve().parent
    candidates.append(package_dir.parent / "ascend-stage-graph")
    path_tool = shutil.which("ascend-stage-graph")
    if path_tool:
        candidates.append(pathlib.Path(path_tool))
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _run_mlir_stage_graph_tool(
    *,
    tool: pathlib.Path,
    stage: dict[str, Any],
    source_path: pathlib.Path,
    output_path: pathlib.Path,
) -> dict[str, Any] | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(tool),
        str(source_path),
        "--stage-order",
        str(stage["order"]),
        "--stage-name",
        str(stage["name"]),
        "--stage-path",
        str(stage["path"]),
        "--output",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (OSError, subprocess.CalledProcessError):
        output_path.unlink(missing_ok=True)
        return None
    try:
        import json

        return json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _edge_class_for_kind(kind: str | None) -> str:
    if not kind:
        return ""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", kind).replace("_", "-").strip("-")
    return f" graph-edge-{safe}" if safe else ""


def _href(from_rel_path: str, to_rel_path: str) -> str:
    import posixpath

    source_dir = pathlib.PurePosixPath(from_rel_path).parent
    return posixpath.relpath(to_rel_path, start=str(source_dir))


def _split_top_level_commas(text: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    angle_depth = 0
    bracket_depth = 0
    paren_depth = 0
    brace_depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth = max(0, angle_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
        elif (
            char == ","
            and not angle_depth
            and not bracket_depth
            and not paren_depth
            and not brace_depth
        ):
            pieces.append(text[start:index].strip())
            start = index + 1
    pieces.append(text[start:].strip())
    return [piece for piece in pieces if piece]


def _collect_func_header(lines: list[str], index: int) -> tuple[str | None, int]:
    if not FUNC_START_RE.search(lines[index]):
        return None, index + 1
    collected: list[str] = []
    cursor = index
    paren_depth = 0
    seen_args = False
    while cursor < len(lines):
        line = lines[cursor]
        collected.append(line)
        for char in line:
            if char == "(":
                seen_args = True
                paren_depth += 1
            elif char == ")" and seen_args:
                paren_depth -= 1
                if paren_depth <= 0:
                    return "\n".join(collected), cursor + 1
        cursor += 1
    return "\n".join(collected), cursor


def _parse_func_decl(header: str) -> tuple[str, list[dict[str, Any]]] | None:
    match = FUNC_START_RE.search(header)
    if not match:
        return None
    args_start = header.find("(", match.start())
    if args_start < 0:
        return match.group("name"), []
    depth = 0
    args_end = -1
    for index, char in enumerate(header[args_start:], start=args_start):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                args_end = index
                break
    if args_end < 0:
        return match.group("name"), []
    args_text = header[args_start + 1 : args_end]
    args = []
    for piece in _split_top_level_commas(args_text):
        arg_match = re.match(
            r"\s*(?P<name>%[A-Za-z0-9_.$-]+)\s*:\s*(?P<type>.+?)\s*$",
            piece,
            re.S,
        )
        if arg_match:
            args.append(
                {
                    "name": arg_match.group("name"),
                    "type": " ".join(arg_match.group("type").split()),
                }
            )
    return match.group("name"), args


def _extract_attr(text: str, name: str) -> str | None:
    match = re.search(rf"{re.escape(name)}\s*=\s*\"([^\"]+)\"", text)
    return match.group(1) if match else None


def _extract_int_attr(text: str, name: str) -> int | None:
    match = re.search(rf"{re.escape(name)}\s*=\s*([0-9]+)", text)
    return int(match.group(1)) if match else None


def _extract_string_list_attr(text: str, name: str) -> list[str]:
    match = re.search(rf"{re.escape(name)}\s*=\s*\[([^\]]*)\]", text)
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


def _extract_balanced_attr_value(
    text: str,
    name: str,
    opener: str,
    closer: str,
) -> str | None:
    match = re.search(rf"{re.escape(name)}\s*=\s*{re.escape(opener)}", text)
    if not match:
        return None
    start = match.end() - 1
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    return None


def _extract_field_string(text: str, name: str) -> str | None:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*\"([^\"]+)\"", text)
    return match.group(1) if match else None


def _extract_field_int(text: str, name: str) -> int | None:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*(-?[0-9]+)", text)
    return int(match.group(1)) if match else None


def _extract_field_string_list(text: str, name: str) -> list[str]:
    body = _extract_balanced_attr_value(text, name, "[", "]")
    if body is None:
        return []
    return re.findall(r'"([^"]+)"', body)


def _extract_attr_as_str(text: str, name: str) -> str | None:
    """Extract an attribute as a string, handling both quoted and integer forms."""
    quoted = _extract_attr(text, name)
    if quoted is not None:
        return quoted
    int_val = _extract_int_attr(text, name)
    return str(int_val) if int_val is not None else None


def _extract_tile_params_attr(text: str) -> list[dict[str, Any]]:
    body = _extract_balanced_attr_value(text, "auto_fuse.tiling_infos", "[", "]")
    if body is None:
        return []
    params: list[dict[str, Any]] = []
    for piece in _split_top_level_commas(body):
        record = piece.strip()
        if record.startswith("{") and record.endswith("}"):
            record = record[1:-1]
        param = {
            key: value
            for key, value in {
                "name": _extract_field_string(record, "name"),
                "axis": _extract_field_int(record, "axis"),
                "axis_kind": _extract_field_string(record, "axis_kind"),
                "binding": _extract_field_string(record, "binding"),
                "default": _extract_field_int(record, "default"),
                "upper_bound": _extract_field_int(record, "upper_bound"),
                "extent": _extract_field_int(record, "extent"),
                "roles": _extract_field_string_list(record, "roles"),
                "primitive_uses": _extract_field_string_list(record, "primitive_uses"),
            }.items()
            if value is not None and value != [] and value is not False
        }
        if param:
            params.append(param)
    return params


def _extract_i64_array_attr(text: str, name: str) -> list[int]:
    match = re.search(rf"{re.escape(name)}\s*=\s*array<i64:\s*([^>]+)>", text)
    if not match:
        return []
    values: list[int] = []
    for item in match.group(1).split(","):
        stripped = item.strip()
        if re.fullmatch(r"-?[0-9]+", stripped):
            values.append(int(stripped))
    return values


def _extract_tail_phases(text: str) -> list[str]:
    phases: list[str] = []
    for affected in re.findall(r"affected\s*=\s*\[([^\]]*)\]", text):
        for phase in re.findall(r'"([^"]+)"', affected):
            if phase not in phases:
                phases.append(phase)
    return phases


def _extract_position(text: str) -> dict[str, Any]:
    match = re.search(r"position\s*=\s*<([^>]+)>", text)
    if not match:
        return {}
    raw = match.group(1).strip()
    position: dict[str, Any] = {"raw": raw}
    pieces = [piece.strip() for piece in raw.split(",")]
    if pieces:
        position["kind"] = pieces[0]
    depth_match = re.search(r"depth\s*=\s*(-?[0-9]+)", raw)
    if depth_match:
        position["depth"] = int(depth_match.group(1))
    double_match = re.search(r"is_double_buffer\s*=\s*(true|false)", raw)
    if double_match:
        position["is_double_buffer"] = double_match.group(1) == "true"
    return position


def _extract_tensor_buffer_attrs(text: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for name in ("tensor_id", "reuse_id", "position_id"):
        value = _extract_int_attr(text, name)
        if value is not None:
            attrs[name] = value
    position = _extract_position(text)
    if position:
        attrs["position"] = position
    memory_space_match = re.search(r"memory_space\s*=\s*([^,}\n]+)", text)
    if memory_space_match:
        attrs["memory_space"] = memory_space_match.group(1).strip()
    return attrs


def _build_semantic_attrs(op_name: str, op_text: str) -> dict[str, Any]:
    kernel = {
        key: value
        for key, value in {
            "id": _extract_attr_as_str(op_text, "auto_fuse.group_id")
            or _extract_attr_as_str(op_text, "auto_fuse.topo_index"),
            "role": _extract_attr(op_text, "auto_fuse.kind")
            or _extract_attr(op_text, "aclnn.op")
            or _extract_attr(op_text, "aclnn.kind"),
            "template_families": _extract_string_list_attr(
                op_text, "afir.reduce_template"
            ),
        }.items()
        if value not in (None, [], False)
    }
    schedule = {
        key: value
        for key, value in {
            "family": _extract_attr(op_text, "afir.reduce_template"),
            "default_tile_size": _extract_attr(op_text, "auto_fuse.default_tile_size"),
            "block_dim": _extract_attr(op_text, "afir.block_dim_expr"),
            "axis_extent": _extract_attr(op_text, "afir.axis_extent_expr"),
            "tile_params": _extract_tile_params_attr(op_text),
        }.items()
        if value not in (None, [], False)
    }
    phases = _extract_tail_phases(op_text)
    if op_name == "memref.copy" and "data_copy" not in phases:
        phases.append("data_copy")
    if op_name.startswith("ascendc.data_copy") and "data_copy" not in phases:
        phases.append("data_copy")
    movement = {"phases": phases} if phases else {}
    memory = _extract_tensor_buffer_attrs(op_text)
    return {
        key: value
        for key, value in {
            "kernel": kernel,
            "schedule": schedule,
            "movement": movement,
            "memory": memory,
        }.items()
        if value
    }


def _build_node_badges(semantic_attrs: dict[str, Any]) -> list[str]:
    badges: list[str] = []
    kernel = semantic_attrs.get("kernel", {})
    schedule = semantic_attrs.get("schedule", {})
    movement = semantic_attrs.get("movement", {})
    memory = semantic_attrs.get("memory", {})
    phases = movement.get("phases")
    if isinstance(phases, list) and phases:
        badges.append("movement phases")
    position = memory.get("position")
    if isinstance(position, dict) and position.get("kind"):
        badge = f"buf {position['kind']}"
        if position.get("depth") is not None:
            badge += f" d{position['depth']}"
        badges.append(badge)
        if position.get("is_double_buffer"):
            badges.append("dbuf")
    tail_policies = schedule.get("tail_policies")
    if isinstance(tail_policies, list) and tail_policies:
        unique_tail = []
        for policy in tail_policies:
            if policy not in unique_tail:
                unique_tail.append(policy)
        badges.append("tail " + "/".join(unique_tail))
    tile_params = schedule.get("tile_params")
    if isinstance(tile_params, list) and tile_params:
        names = [
            str(param.get("name"))
            for param in tile_params
            if isinstance(param, dict) and param.get("name")
        ]
        badges.append("tile " + "/".join(names) if names else "tile params")
    if kernel.get("role"):
        badges.append(str(kernel["role"]))
    return badges


def _extract_result_type(text: str) -> str | None:
    arrow_matches = list(re.finditer(r"->\s*([^\n{]+)", text))
    if arrow_matches:
        return arrow_matches[-1].group(1).strip()
    colon_match = re.search(r":\s*([^\n]+)$", text.strip())
    return colon_match.group(1).strip() if colon_match else None


def _collect_op_text(lines: list[str], index: int, op_name: str) -> tuple[str, int]:
    collected = [lines[index]]
    if op_name == "linalg.generic":
        seen_region = lines[index].strip().startswith("^bb")
        cursor = index + 1
        while cursor < len(lines):
            collected.append(lines[cursor])
            stripped = lines[cursor].strip()
            if stripped.startswith("^bb"):
                seen_region = True
            if seen_region and stripped.startswith("}"):
                cursor += 1
                break
            cursor += 1
        return "\n".join(collected), cursor
    return lines[index], index + 1


def _match_resultless_op(line: str) -> re.Match[str] | None:
    stripped = line.strip()
    if (
        not stripped
        or stripped.startswith(("#", "//", "^", "}"))
        or stripped.startswith("module")
        or stripped.startswith("func.func")
    ):
        return None
    match = RESULTLESS_OP_RE.match(line)
    if not match:
        return None
    op_name = match.group("op")
    return match if "." in op_name else None


def _extract_linalg_out_values(op_text: str) -> list[str]:
    match = re.search(r"\bouts\((?P<body>.*?)\)", op_text, re.S)
    if not match:
        return []
    values: list[str] = []
    for value in SSA_VALUE_RE.findall(match.group("body")):
        if value not in values:
            values.append(value)
    return values


def _observes_memory_effect(op_name: str) -> bool:
    if op_name == "func.return" or op_name.startswith("linalg."):
        return True
    return op_name in {
        "affine.load",
        "affine.store",
        "memref.atomic_rmw",
        "memref.copy",
        "memref.generic_atomic_rmw",
        "memref.load",
        "memref.store",
        "vector.transfer_read",
        "vector.transfer_write",
    }


RESOURCE_RULES: dict[str, dict[str, Any]] = {
    "ascendc.pipe.init_queue": {
        "read_inputs": (0,),
        "write_inputs": (1,),
    },
    "ascendc.pipe.init_buffer": {
        "read_inputs": (0,),
        "write_inputs": (1,),
    },
    "ascendc.que_bind.alloc_tensor": {
        "read_inputs": (0,),
        "write_inputs": (0,),
        "write_results": True,
    },
    "ascendc.que_bind.deque_tensor": {
        "read_inputs": (0,),
        "write_inputs": (0,),
        "write_results": True,
    },
    "ascendc.que_bind.enque_tensor": {
        "read_inputs": (1,),
        "write_inputs": (0,),
    },
    "ascendc.que_bind.free_tensor": {
        "read_inputs": (1,),
        "write_inputs": (0,),
    },
    "ascendc.global_tensor.set_global_buffer": {
        "read_inputs_from": 1,
        "write_inputs": (0,),
    },
    "ascendc.add_l2": {
        "read_inputs_from": 1,
        "write_inputs": (0,),
    },
    "ascendc.broadcast_l2": {
        "read_inputs_from": 1,
        "write_inputs": (0,),
    },
    "ascendc.mul_l2": {
        "read_inputs_from": 1,
        "write_inputs": (0,),
    },
    "emitasc.member": {
        "read_inputs": (0,),
        "write_results": True,
    },
    "emitasc.verbatim": {
        "read_inputs_from": 1,
        "write_inputs": (0,),
    },
}

TERMINAL_OP_REASONS: dict[str, str] = {
    "affine.apply": "dead-helper-value",
    "arith.addi": "dead-helper-value",
    "arith.constant": "constant",
    "arith.index_cast": "dead-helper-value",
    "ascendc.que_bind.free_tensor": "resource-free-terminal",
    "emitasc.declare_py_struct": "emit-declaration",
    "emitasc.member": "emit-member-read",
    "emitasc.verbatim": "emit-verbatim-side-effect",
    "func.arg": "unused-argument",
    "func.return": "function-return",
    "linalg.yield": "region-yield",
    "memref.cast": "dead-helper-value",
    "memref.dim": "dead-helper-value",
    "return": "function-return",
    "scf.for": "structured-control",
    "scf.if": "structured-control",
}


PURE_VALUE_RESOURCE_OPS = {
    "ascendc.get_block_idx",
    "ascendc.global_tensor",
    "ascendc.pipe",
    "ascendc.queue",
    "ascendc.tbuf",
    "emitasc.copy_struct",
    "emitasc.member",
    "emitasc.reinterpret_cast",
}


def _classify_resource_access(
    op_name: str,
    input_values: list[str],
    result_values: list[str],
) -> tuple[list[str], list[str], bool]:
    reads: list[str] = []
    writes: list[str] = []

    def add_read(value: str) -> None:
        if value not in reads:
            reads.append(value)

    def add_write(value: str) -> None:
        if value not in writes:
            writes.append(value)

    def add_reads(values: list[str]) -> None:
        for value in values:
            add_read(value)

    if op_name == "ascendc.pipe_barrier":
        return [], [], True
    rule = RESOURCE_RULES.get(op_name)
    if rule is not None:
        for index in rule.get("read_inputs", ()):
            if index < len(input_values):
                add_read(input_values[index])
        for index in rule.get("write_inputs", ()):
            if index < len(input_values):
                add_write(input_values[index])
        if "read_inputs_from" in rule:
            add_reads(input_values[rule["read_inputs_from"] :])
        if rule.get("write_results"):
            for value in result_values:
                add_write(value)
        return reads, writes, False
    if op_name.startswith("ascendc.") and input_values:
        add_reads(input_values[1:])
        add_write(input_values[0])
    return reads, writes, False


def _extract_region_body(op_text: str) -> str | None:
    body_lines: list[str] = []
    in_region = False
    for line in op_text.splitlines()[1:]:
        stripped = line.strip()
        if stripped.startswith("^bb"):
            in_region = True
        if in_region and stripped.startswith("}"):
            break
        if in_region and stripped:
            body_lines.append(stripped)
    return "\n".join(body_lines) if body_lines else None


def _extract_body_ops(region_body: str | None) -> list[str]:
    if not region_body:
        return []
    body_ops: list[str] = []
    for line in region_body.splitlines():
        if line.strip().startswith("^"):
            continue
        match = BODY_OP_RE.match(line)
        if match:
            body_ops.append(match.group("op"))
    return body_ops


def _summarize_body_ops(body_ops: list[str]) -> str | None:
    compute_ops = [op for op in body_ops if op != "linalg.yield"]
    if compute_ops:
        return " -> ".join(compute_ops)
    if body_ops:
        return " -> ".join(body_ops)
    return None


def parse_stage_mlir(stage: dict[str, Any], text: str) -> dict[str, Any]:
    lines = text.splitlines()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    producer_by_value: dict[str, str] = {}
    alias_base_by_value: dict[str, str] = {}
    memory_writers_by_base: dict[str, list[str]] = {}
    memory_effect_edges: set[tuple[str, str, str]] = set()
    resource_writer_by_value: dict[str, str] = {}
    resource_effect_edges: set[tuple[str, str, str, str]] = set()
    control_edges: set[tuple[str, str, str]] = set()
    last_effect_node: str | None = None
    pending_barrier_node: str | None = None
    defined_values: set[str] = set()
    function_names: list[str] = []
    function_node_ids: dict[str, list[str]] = {}

    def alias_base(value: str) -> str:
        seen: set[str] = set()
        current = value
        while current in alias_base_by_value and current not in seen:
            seen.add(current)
            current = alias_base_by_value[current]
        return current

    def append_edge(
        source: str,
        target: str,
        value: str,
        *,
        kind: str | None = None,
        effect: str | None = None,
    ) -> None:
        edge: dict[str, Any] = {
            "id": f"e{len(edges)}",
            "from": source,
            "to": target,
            "value": value,
        }
        if kind is not None:
            edge["kind"] = kind
        if effect is not None:
            edge["effect"] = effect
        edges.append(edge)

    def append_control_edge(source: str, target: str, value: str, effect: str) -> None:
        key = (source, target, value)
        if key in control_edges:
            return
        control_edges.add(key)
        append_edge(source, target, value, kind="control", effect=effect)

    def note_function(name: str) -> None:
        if name in function_node_ids:
            return
        function_names.append(name)
        function_node_ids[name] = []

    index = 0
    while index < len(lines):
        line_number = index + 1
        func_header, next_index = _collect_func_header(lines, index)
        if not func_header:
            index = next_index
            continue
        func_decl = _parse_func_decl(func_header)
        if not func_decl:
            index = next_index
            continue
        function_name, args = func_decl
        note_function(function_name)
        for arg in args:
            node_id = f"n{len(nodes)}"
            node = {
                "id": node_id,
                "line": line_number,
                "function": function_name,
                "op_name": "func.arg",
                "label": arg["name"],
                "input_values": [],
                "result_values": [arg["name"]],
                "result_type": arg["type"],
                "kernel_id": None,
                "op_role": None,
                "schedule_decision_id": None,
                "workspace_size_bytes": None,
                "semantic_attrs": {},
                "badges": [],
            }
            nodes.append(node)
            function_node_ids[function_name].append(node_id)
            producer_by_value[arg["name"]] = node_id
            defined_values.add(arg["name"])
        index = next_index

    index = 0
    current_function = None
    while index < len(lines):
        line = lines[index]
        func_header, _ = _collect_func_header(lines, index)
        func_decl = _parse_func_decl(func_header) if func_header else None
        if func_decl:
            current_function = func_decl[0]
            note_function(current_function)
        op_match = OP_RE.match(line)
        return_match = RETURN_RE.match(line)
        resultless_match = None if op_match or return_match else _match_resultless_op(line)
        if not op_match and not return_match and not resultless_match:
            index += 1
            continue

        if op_match:
            op_name = op_match.group("op")
            op_text, next_index = _collect_op_text(lines, index, op_name)
            result_values = [value.strip() for value in op_match.group("results").split(",")]
            raw_values = SSA_VALUE_RE.findall(op_text)
            input_values = []
            for value in raw_values:
                if value in result_values or value not in defined_values:
                    continue
                if value not in input_values:
                    input_values.append(value)
        elif resultless_match:
            op_name = resultless_match.group("op")
            op_text, next_index = _collect_op_text(lines, index, op_name)
            result_values = []
            input_values = []
            for value in SSA_VALUE_RE.findall(op_text):
                if value in defined_values and value not in input_values:
                    input_values.append(value)
        else:
            op_name = "func.return"
            op_text = line
            next_index = index + 1
            result_values = []
            input_values = []
            for value in SSA_VALUE_RE.findall(line):
                if value in defined_values and value not in input_values:
                    input_values.append(value)

        node_id = f"n{len(nodes)}"
        region_body = _extract_region_body(op_text)
        body_ops = _extract_body_ops(region_body)
        semantic_attrs = _build_semantic_attrs(op_name, op_text)
        node = {
            "id": node_id,
            "line": index + 1,
            "line_end": next_index,
            "function": current_function,
            "op_name": op_name,
            "label": result_values[0] if result_values else op_name,
            "input_values": input_values,
            "result_values": result_values,
            "result_type": _extract_result_type(op_text),
            "source_excerpt": op_text,
            "region_body": region_body,
            "body_ops": body_ops,
            "body_summary": _summarize_body_ops(body_ops),
            "kernel_id": _extract_attr_as_str(op_text, "auto_fuse.group_id"),
            "op_role": _extract_attr(op_text, "auto_fuse.kind"),
            "schedule_decision_id": None,
            "workspace_size_bytes": _extract_int_attr(op_text, "cann.workspace_size_bytes"),
            "semantic_attrs": semantic_attrs,
            "badges": _build_node_badges(semantic_attrs),
        }
        nodes.append(node)
        if current_function:
            function_node_ids.setdefault(current_function, []).append(node_id)

        for value in input_values:
            producer = producer_by_value.get(value)
            if producer:
                append_edge(producer, node_id, value)
        memory_bases = []
        for value in input_values:
            base = alias_base(value)
            if base not in memory_bases:
                memory_bases.append(base)
        if _observes_memory_effect(op_name):
            for base in memory_bases:
                for writer in memory_writers_by_base.get(base, []):
                    if writer == node_id:
                        continue
                    key = (writer, node_id, base)
                    if key in memory_effect_edges:
                        continue
                    memory_effect_edges.add(key)
                    append_edge(writer, node_id, base, kind="memory_effect", effect="write")
        for value in result_values:
            producer_by_value[value] = node_id
            defined_values.add(value)
        if op_name in ("memref.subview", "memref.cast", "memref.reinterpret_cast") and input_values:
            base = alias_base(input_values[0])
            for value in result_values:
                alias_base_by_value[value] = base
        written_bases = []
        if op_name == "memref.copy" and len(input_values) >= 2:
            written_bases.append(alias_base(input_values[1]))
        if op_name.startswith("linalg."):
            written_bases.extend(alias_base(value) for value in _extract_linalg_out_values(op_text))
        for base in written_bases:
            writers = memory_writers_by_base.setdefault(base, [])
            if node_id not in writers:
                writers.append(node_id)
        resource_reads, resource_writes, is_barrier = _classify_resource_access(
            op_name, input_values, result_values
        )
        has_resource_access = bool(resource_reads or resource_writes)
        if is_barrier:
            if last_effect_node and last_effect_node != node_id:
                append_control_edge(last_effect_node, node_id, "pipe_all", "barrier")
            pending_barrier_node = node_id
            last_effect_node = node_id
        elif has_resource_access:
            if pending_barrier_node and pending_barrier_node != node_id:
                append_control_edge(pending_barrier_node, node_id, "pipe_all", "barrier")
                pending_barrier_node = None
            for value in resource_reads + resource_writes:
                writer = resource_writer_by_value.get(value)
                if not writer or writer == node_id:
                    continue
                effect = "write" if value in resource_writes else "read"
                key = (writer, node_id, value, effect)
                if key in resource_effect_edges:
                    continue
                resource_effect_edges.add(key)
                append_edge(writer, node_id, value, kind="resource_effect", effect=effect)
            for value in resource_writes:
                resource_writer_by_value[value] = node_id
            last_effect_node = node_id
        elif written_bases:
            last_effect_node = node_id
        index = next_index

    kernel_ids = sorted(
        {node["kernel_id"] for node in nodes if isinstance(node.get("kernel_id"), str)}
    )
    graph = {
        "schema_version": 1,
        "tool": "ascend-debug",
        "stage": {
            "order": stage["order"],
            "name": stage["name"],
            "path": stage["path"],
        },
        "function": function_names[0] if function_names else None,
        "functions": [
            {"name": name, "node_ids": function_node_ids.get(name, [])}
            for name in function_names
            if function_node_ids.get(name)
        ],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "kernel_count": len(kernel_ids),
        "kernels": [
            {
                "kernel_id": kernel_id,
                "node_ids": [node["id"] for node in nodes if node.get("kernel_id") == kernel_id],
            }
            for kernel_id in kernel_ids
        ],
        "nodes": nodes,
        "edges": edges,
    }
    return _finalize_graph(graph)


def _edge_rows(graph: dict[str, Any]) -> str:
    node_labels = {node["id"]: node.get("label") or node.get("op_name") for node in graph["nodes"]}
    rows = []
    for edge in graph["edges"]:
        rows.append(
            "<tr>"
            f"<td>{_cell(edge.get('value'))}</td>"
            f"<td>{_cell(node_labels.get(edge.get('from')))}</td>"
            f"<td>{_cell(node_labels.get(edge.get('to')))}</td>"
            "</tr>"
        )
    return "\n".join(rows) or '<tr><td colspan="3">No data-flow edges detected.</td></tr>'


def _kernel_rows(graph: dict[str, Any]) -> str:
    rows = []
    for kernel in graph.get("kernels", []):
        rows.append(
            "<tr>"
            f"<td>{_cell(kernel.get('kernel_id'))}</td>"
            f"<td>{_cell(len(kernel.get('node_ids', [])))}</td>"
            f"<td>{_cell(', '.join(kernel.get('node_ids', [])))}</td>"
            "</tr>"
        )
    return "\n".join(rows) or '<tr><td colspan="3">No kernel boundary in this stage.</td></tr>'


def _node_brief(node: dict[str, Any], reason: str | None = None) -> dict[str, Any]:
    brief = {
        "id": node.get("id"),
        "line": node.get("line"),
        "op_name": node.get("op_name"),
        "label": node.get("label"),
    }
    if reason:
        brief["reason"] = reason
    return brief


def _terminal_reason(node: dict[str, Any], out_count: int, in_count: int) -> str | None:
    op_name = str(node.get("op_name") or "")
    if out_count > 0:
        return None
    if op_name in TERMINAL_OP_REASONS:
        return TERMINAL_OP_REASONS[op_name]
    if op_name in PURE_VALUE_RESOURCE_OPS:
        return "dead-resource-value"
    if op_name.startswith("arith."):
        return "dead-helper-value"
    if op_name.startswith("emitasc.declare"):
        return "emit-declaration"
    if op_name.startswith("emitasc."):
        return "emit-terminal"
    if op_name.startswith("scf.") and in_count > 0:
        return "structured-control"
    return None


def _is_effect_like_node(node: dict[str, Any]) -> bool:
    op_name = str(node.get("op_name") or "")
    if op_name in PURE_VALUE_RESOURCE_OPS:
        return False
    if op_name in RESOURCE_RULES or op_name == "ascendc.pipe_barrier":
        return True
    if op_name.startswith("ascendc.") and not node.get("result_values"):
        return True
    if op_name.startswith("emitasc.") and not node.get("result_values"):
        return True
    return (
        op_name in {"affine.store", "memref.copy", "memref.store", "vector.transfer_write"}
        or op_name.startswith("linalg.")
    )


def _compute_connectivity(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_ids = {node.get("id") for node in nodes}
    in_count: dict[str, int] = {node["id"]: 0 for node in nodes if isinstance(node.get("id"), str)}
    out_count: dict[str, int] = {node["id"]: 0 for node in nodes if isinstance(node.get("id"), str)}
    typed_count: dict[str, int] = {node["id"]: 0 for node in nodes if isinstance(node.get("id"), str)}
    adjacency: dict[str, set[str]] = {node["id"]: set() for node in nodes if isinstance(node.get("id"), str)}
    edge_kind_counts: dict[str, int] = {}

    for edge in edges:
        source = edge.get("from")
        target = edge.get("to")
        kind = edge.get("kind") or "value"
        edge_kind_counts[kind] = edge_kind_counts.get(kind, 0) + 1
        if source not in node_ids or target not in node_ids:
            continue
        if isinstance(source, str):
            out_count[source] = out_count.get(source, 0) + 1
        if isinstance(target, str):
            in_count[target] = in_count.get(target, 0) + 1
        if isinstance(source, str) and isinstance(target, str):
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)
            if kind in {"control", "memory_effect", "resource_effect", "region", "symbol"}:
                typed_count[source] = typed_count.get(source, 0) + 1
                typed_count[target] = typed_count.get(target, 0) + 1

    seen: set[str] = set()
    component_sizes: list[int] = []
    for node_id in adjacency:
        if node_id in seen:
            continue
        stack = [node_id]
        seen.add(node_id)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency.get(current, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        component_sizes.append(size)

    isolated_nodes = []
    suspicious_isolated_nodes = []
    allowed_terminal_nodes = []
    dangling_effect_nodes = []
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        incoming = in_count.get(node_id, 0)
        outgoing = out_count.get(node_id, 0)
        terminal_reason = _terminal_reason(node, outgoing, incoming)
        if incoming == 0 and outgoing == 0:
            isolated_nodes.append(_node_brief(node, terminal_reason))
            if terminal_reason is None:
                suspicious_isolated_nodes.append(_node_brief(node))
        if terminal_reason is not None:
            allowed_terminal_nodes.append(_node_brief(node, terminal_reason))
        if _is_effect_like_node(node) and typed_count.get(node_id, 0) == 0 and terminal_reason is None:
            dangling_effect_nodes.append(_node_brief(node))

    return {
        "component_count": len(component_sizes),
        "component_sizes": sorted(component_sizes, reverse=True),
        "isolated_count": len(isolated_nodes),
        "isolated_nodes": isolated_nodes,
        "suspicious_isolated_count": len(suspicious_isolated_nodes),
        "suspicious_isolated_nodes": suspicious_isolated_nodes,
        "dangling_effect_count": len(dangling_effect_nodes),
        "dangling_effect_nodes": dangling_effect_nodes,
        "allowed_terminal_count": len(allowed_terminal_nodes),
        "allowed_terminal_nodes": allowed_terminal_nodes,
        "edge_kind_counts": edge_kind_counts,
    }


def _finalize_graph(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.setdefault("nodes", [])
    edges = graph.setdefault("edges", [])
    for index, edge in enumerate(edges):
        edge.setdefault("id", f"e{index}")
        edge.setdefault("kind", "value")
        if edge.get("label") is None:
            effect = edge.get("effect")
            value = edge.get("value", "")
            edge["label"] = f"{effect} {value}" if effect else value
    kernel_ids = sorted(
        {
            node.get("kernel_id")
            for node in nodes
            if isinstance(node.get("kernel_id"), str)
        }
    )
    graph["node_count"] = len(nodes)
    graph["edge_count"] = len(edges)
    graph["kernel_count"] = len(kernel_ids)
    graph["kernels"] = [
        {
            "kernel_id": kernel_id,
            "node_ids": [node["id"] for node in nodes if node.get("kernel_id") == kernel_id],
        }
        for kernel_id in kernel_ids
    ]
    graph["kernel_ids"] = kernel_ids
    graph["connectivity"] = _compute_connectivity(graph)
    return graph


def _truncate(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def _compute_graph_layout(graph: dict[str, Any]) -> dict[str, Any]:
    node_width = 220
    node_height = 120
    column_gap = 72
    layer_gap = 86
    margin_x = 36
    margin_y = 36
    predecessor_ids: dict[str, list[str]] = {
        node["id"]: []
        for node in graph["nodes"]
    }
    for edge in graph["edges"]:
        predecessor_ids.setdefault(edge["to"], []).append(edge["from"])

    layer_by_id: dict[str, int] = {}
    layers: dict[int, list[str]] = {}
    for node in graph["nodes"]:
        node_id = node["id"]
        predecessors = [
            layer_by_id[pred] + 1
            for pred in predecessor_ids.get(node_id, [])
            if pred in layer_by_id
        ]
        layer = max(predecessors) if predecessors else 0
        layer_by_id[node_id] = layer
        layers.setdefault(layer, []).append(node_id)

    node_layout: dict[str, dict[str, Any]] = {}
    max_x = margin_x
    max_y = margin_y
    for layer in sorted(layers):
        for row, node_id in enumerate(layers[layer]):
            x = margin_x + row * (node_width + column_gap)
            y = margin_y + layer * (node_height + layer_gap)
            node_layout[node_id] = {
                "x": x,
                "y": y,
                "width": node_width,
                "height": node_height,
                "layer": layer,
                "row": row,
            }
            max_x = max(max_x, x + node_width)
            max_y = max(max_y, y + node_height)

    edge_layout = []
    for edge in graph["edges"]:
        source = node_layout.get(edge["from"])
        target = node_layout.get(edge["to"])
        if not source or not target:
            continue
        source_x = source["x"] + source["width"] / 2
        source_y = source["y"] + source["height"]
        target_x = target["x"] + target["width"] / 2
        target_y = target["y"]
        bend = max(42, abs(target_y - source_y) / 2)
        path = (
            f"M {source_x:.1f} {source_y:.1f} "
            f"C {source_x:.1f} {source_y + bend:.1f}, "
            f"{target_x:.1f} {target_y - bend:.1f}, "
            f"{target_x:.1f} {target_y:.1f}"
        )
        edge_layout.append(
            {
                "id": edge["id"],
                "from": edge["from"],
                "to": edge["to"],
                "value": edge["value"],
                "kind": edge.get("kind"),
                "effect": edge.get("effect"),
                "label": edge.get("label")
                or (
                    f"{edge.get('effect')} {edge['value']}"
                    if edge.get("effect")
                    else edge["value"]
                ),
                "path": path,
                "label_x": (source_x + target_x) / 2,
                "label_y": (source_y + target_y) / 2 - 8,
            }
        )

    return {
        "visual_kind": "svg-dag",
        "direction": "top-to-bottom",
        "node_width": node_width,
        "node_height": node_height,
        "width": max(720, max_x + margin_x),
        "height": max(420, max_y + margin_y),
        "nodes": node_layout,
        "edges": edge_layout,
    }


def render_svg_graph(
    graph: dict[str, Any],
    *,
    svg_id: str = "stage-graph-svg",
    canvas_class: str = "graph-canvas",
) -> str:
    graph_layout = graph.get("layout", {})
    node_layout = graph_layout.get("nodes", {})
    edge_layout = graph_layout.get("edges", [])
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    node_index_by_id = {
        node["id"]: index
        for index, node in enumerate(graph["nodes"])
    }
    edge_elements = []
    for edge in edge_layout:
        edge_class = _edge_class_for_kind(edge.get("kind"))
        edge_elements.append(
            f'<g class="graph-edge{edge_class}">'
            f'<path class="graph-edge-path{edge_class}" d="{_cell(edge.get("path"))}" />'
            f'<text class="graph-edge-label" x="{_cell(edge.get("label_x"))}" '
            f'y="{_cell(edge.get("label_y"))}">{_cell(_truncate(edge.get("label", edge.get("value")), 24))}</text>'
            "</g>"
        )

    node_elements = []
    for node_id, position in node_layout.items():
        node = node_by_id.get(node_id)
        if not node:
            continue
        index = node_index_by_id[node_id]
        result = ", ".join(node.get("result_values", [])) or node.get("label")
        inputs = ", ".join(node.get("input_values", [])) or "root"
        detail = f"body: {node.get('body_summary')}" if node.get("body_summary") else f"in: {inputs}"
        kernel = node.get("kernel_id")
        classes = "graph-node kernel-node" if kernel else "graph-node"
        kernel_text = f"kernel {_truncate(kernel, 22)}" if kernel else f"line {node.get('line')}"
        node_elements.append(
            f'<g class="{classes}" data-node-index="{index}" tabindex="0" role="button" '
            f'transform="translate({_cell(position.get("x"))},{_cell(position.get("y"))})">'
            f'<title>{_cell(node.get("op_name"))}</title>'
            f'<rect width="{_cell(position.get("width"))}" '
            f'height="{_cell(position.get("height"))}" rx="6" />'
            f'<text class="node-op" x="14" y="24">{_cell(_truncate(node.get("op_name"), 28))}</text>'
            f'<text class="node-result" x="14" y="47">{_cell(_truncate(result, 30))}</text>'
            f'<text class="node-inputs" x="14" y="68">{_cell(_truncate(detail, 30))}</text>'
            f'<text class="node-kernel" x="206" y="22">{_cell(kernel_text)}</text>'
            "</g>"
        )

    if not node_elements:
        node_elements.append('<text x="24" y="42">No graph nodes detected.</text>')

    return f"""
<div class="{_cell(canvas_class)}" aria-label="Stage data-flow graph">
<svg id="{_cell(svg_id)}" width="{_cell(graph_layout.get('width', 720))}"
     height="{_cell(graph_layout.get('height', 420))}"
     viewBox="0 0 {_cell(graph_layout.get('width', 720))} {_cell(graph_layout.get('height', 420))}"
     xmlns="http://www.w3.org/2000/svg">
<defs>
<marker id="arrow-head" viewBox="0 0 10 10" refX="9" refY="5"
        markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" />
</marker>
</defs>
{''.join(edge_elements)}
{''.join(node_elements)}
</svg>
</div>
"""


def _render_stage_graph_html(
    *,
    run_dir: pathlib.Path,
    graph: dict[str, Any],
    view_rel_path: str,
    raw_mlir_rel_path: str,
    stage_links: list[dict[str, str]],
) -> None:
    dashboard_href = html.escape(_href(view_rel_path, "index.html"), quote=True)
    raw_href = html.escape(_href(view_rel_path, raw_mlir_rel_path), quote=True)
    graph_json_rel, _ = _stage_graph_rel_paths(raw_mlir_rel_path)
    json_href = html.escape(_href(view_rel_path, graph_json_rel), quote=True)
    timeline_links = []
    current_stage = graph["stage"]["path"]
    for link in stage_links:
        active = " active" if link["stage_path"] == current_stage else ""
        href = html.escape(_href(view_rel_path, link["view_rel_path"]), quote=True)
        timeline_links.append(
            f'<a class="stage-link{active}" href="{href}">{_cell(link["label"])}</a>'
        )
    svg_graph = render_svg_graph(graph)
    nodes_json = layout.json_script_payload(graph["nodes"])
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_cell(graph['stage']['name'])} Stage Graph - ascend-debug</title>
<style>
body {{ margin: 0; font-family: sans-serif; color: #17202a; background: #f8fafc; }}
header {{ padding: 0.85rem 1rem; background: #ffffff; border-bottom: 1px solid #cbd5e1; position: sticky; top: 0; z-index: 2; }}
h1 {{ margin: 0 0 0.35rem; font-size: 1rem; }}
.toolbar {{ display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; }}
.shell {{ display: grid; grid-template-columns: 13rem minmax(28rem, 1fr) 22rem; min-height: calc(100vh - 4.25rem); }}
.timeline {{ border-right: 1px solid #cbd5e1; background: #ffffff; padding: 0.85rem; }}
.stage-link {{ display: block; padding: 0.45rem 0.55rem; border-radius: 5px; color: #334155; text-decoration: none; }}
.stage-link.active {{ background: #dbeafe; color: #1d4ed8; font-weight: 700; }}
.canvas {{ padding: 1rem; overflow: auto; }}
.summary {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.75rem; }}
.badge {{ display: inline-block; margin-left: 0.4rem; padding: 0.08rem 0.35rem; border: 1px solid #cbd5e1; border-radius: 999px; background: #f8fafc; color: #475569; font-size: 0.75rem; }}
.badge.kernel {{ border-color: #93c5fd; background: #eff6ff; color: #1d4ed8; }}
.graph-canvas {{ width: 100%; overflow: auto; border: 1px solid #cbd5e1; border-radius: 6px; background: #ffffff; }}
#stage-graph-svg {{ display: block; min-width: 100%; }}
.graph-edge-path {{ fill: none; stroke: #64748b; stroke-width: 1.5; marker-end: url(#arrow-head); }}
.graph-edge-path.graph-edge-value {{ stroke: #64748b; }}
.graph-edge-path.graph-edge-memory-effect {{ stroke: #0f766e; stroke-dasharray: 6 4; }}
.graph-edge-path.graph-edge-resource-effect {{ stroke: #7c3aed; stroke-dasharray: 6 4; }}
.graph-edge-path.graph-edge-control {{ stroke: #ea580c; stroke-dasharray: 2 4; }}
.graph-edge-path.graph-edge-region {{ stroke: #60a5fa; stroke-dasharray: 2 6; opacity: 0.72; }}
.graph-edge-path.graph-edge-symbol {{ stroke: #475569; stroke-dasharray: 4 4; }}
.graph-edge-label {{ fill: #475569; font-size: 11px; font-family: SFMono-Regular, Menlo, Consolas, monospace; }}
.graph-node {{ cursor: pointer; outline: none; }}
.graph-node rect {{ fill: #ffffff; stroke: #94a3b8; stroke-width: 1.4; }}
.graph-node.kernel-node rect {{ fill: #eff6ff; stroke: #2563eb; }}
.graph-node:hover rect, .graph-node.selected rect {{ stroke: #0f766e; stroke-width: 2.4; }}
.node-op {{ fill: #17202a; font-size: 13px; font-weight: 700; }}
.node-result {{ fill: #334155; font-size: 12px; font-family: SFMono-Regular, Menlo, Consolas, monospace; }}
.node-inputs {{ fill: #64748b; font-size: 11px; font-family: SFMono-Regular, Menlo, Consolas, monospace; }}
.node-kernel {{ fill: #1d4ed8; font-size: 10px; text-anchor: end; }}
.inspector {{ border-left: 1px solid #cbd5e1; background: #ffffff; padding: 0.85rem; overflow: auto; }}
table {{ width: 100%; border-collapse: collapse; background: #ffffff; margin-top: 0.75rem; }}
th, td {{ border: 1px solid #cbd5e1; padding: 0.35rem 0.45rem; text-align: left; vertical-align: top; }}
th {{ background: #f1f5f9; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #0b1020; color: #dbeafe; border: 1px solid #1e293b; padding: 0.75rem; border-radius: 6px; }}
</style>
</head>
<body>
<header>
<h1>Stage Graph: {_cell(graph['stage']['name'])}</h1>
<div class="toolbar">
<a href="{dashboard_href}">Dashboard</a>
<a href="{raw_href}">Raw MLIR</a>
<a href="{json_href}">Graph JSON</a>
</div>
</header>
<main class="shell">
<aside class="timeline">
<strong>Stages</strong>
{''.join(timeline_links)}
</aside>
<section class="canvas">
<h2>Stage Graph</h2>
<div class="summary">
<span class="badge">nodes {_cell(graph['node_count'])}</span>
<span class="badge">edges {_cell(graph['edge_count'])}</span>
<span class="badge">kernels {_cell(graph['kernel_count'])}</span>
</div>
{svg_graph}
<h2>Kernel boundary</h2>
<table>
<thead><tr><th>Kernel</th><th>Nodes</th><th>Node ids</th></tr></thead>
<tbody>{_kernel_rows(graph)}</tbody>
</table>
<h2>Data edges</h2>
<table>
<thead><tr><th>Value</th><th>Producer</th><th>Consumer</th></tr></thead>
<tbody>{_edge_rows(graph)}</tbody>
</table>
</section>
<aside class="inspector">
<h2>Inspector</h2>
<p>Select a graph node to inspect compiler-visible facts.</p>
<pre id="inspector-json"></pre>
</aside>
</main>
<script type="application/json" id="nodes-json">{nodes_json}</script>
<script>
const nodes = JSON.parse(document.getElementById("nodes-json").textContent);
const inspector = document.getElementById("inspector-json");
const graphNodes = Array.from(document.querySelectorAll(".graph-node"));
function selectNode(index) {{
  graphNodes.forEach((node) => {{
    node.classList.toggle("selected", Number(node.dataset.nodeIndex) === index);
  }});
  inspector.textContent = JSON.stringify(nodes[index], null, 2);
}}
graphNodes.forEach((node) => {{
  const index = Number(node.dataset.nodeIndex);
  node.addEventListener("click", () => selectNode(index));
  node.addEventListener("keydown", (event) => {{
    if (event.key === "Enter" || event.key === " ") {{
      event.preventDefault();
      selectNode(index);
    }}
  }});
}});
if (graphNodes.length) selectNode(Number(graphNodes[0].dataset.nodeIndex));
</script>
</body>
</html>
"""
    layout.write_text(run_dir / view_rel_path, document)


def render_stage_graphs(run_dir: pathlib.Path, stages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    graph_views: dict[str, dict[str, Any]] = {}
    mlir_tool = _find_mlir_stage_graph_tool()
    for stage in sorted(stages, key=lambda item: item["order"]):
        rel_path = stage["path"]
        source_path = run_dir / rel_path
        if not source_path.exists():
            continue
        json_rel_path, _ = _stage_graph_rel_paths(rel_path)
        graph: dict[str, Any] | None = None
        if mlir_tool is not None:
            graph = _run_mlir_stage_graph_tool(
                tool=mlir_tool,
                stage=stage,
                source_path=source_path,
                output_path=run_dir / json_rel_path,
            )
        if graph is None:
            text = source_path.read_text(encoding="utf-8", errors="replace")
            graph = parse_stage_mlir(stage, text)
            graph["graph_source"] = "python-parser"
        graph = _finalize_graph(graph)
        graph["layout"] = _compute_graph_layout(graph)
        layout.write_json(run_dir / json_rel_path, graph)
        graph_views[rel_path] = {
            "json_rel_path": json_rel_path,
            "node_count": graph["node_count"],
            "edge_count": graph["edge_count"],
            "kernel_count": graph["kernel_count"],
            "graph_source": graph.get("graph_source", "python-parser"),
        }
    return graph_views
