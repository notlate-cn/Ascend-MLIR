# python/test/test_tiling_schema.py
"""
Unit tests for TilingSchema.

Run:
    cd /Volumes/GM9/code/Ascend-MLIR
    python3 -m pytest python/test/test_tiling_schema.py -v

The fixture is examples/broadcast-add-reduce/tiling_space.json.
It has 6 int64 fields: TB_M, TB_N, dim_arg0_0, dim_arg1_1, dim_arg0_1, dim_arg1_0.
"""
import json
import struct
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from python.runtime.tiling_schema import TilingSchema

FIXTURE = (
    Path(__file__).parent.parent.parent
    / "examples/broadcast-add-reduce/tiling_space.json"
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_json(params: list[dict]) -> Path:
    """Write a temporary tiling_space.json and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False
    )
    json.dump({"kernel": "test", "tiling_params": params}, tmp)
    tmp.close()
    return Path(tmp.name)


# ── from_json ─────────────────────────────────────────────────────────────────

def test_from_json_roundtrip():
    schema = TilingSchema.from_json(FIXTURE)
    assert schema.field_names == [
        "TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"
    ]
    assert len(schema) == 6
    assert all(f["type"] == "int64" for f in schema.fields)


def test_from_json_file_not_found():
    with pytest.raises(FileNotFoundError, match="not found"):
        TilingSchema.from_json("/nonexistent/path/schema.json")


def test_from_json_malformed_json():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write("{ not valid json")
        path = Path(f.name)
    with pytest.raises(ValueError, match="JSON parse error"):
        TilingSchema.from_json(path)


def test_from_json_missing_tiling_params():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump({"kernel": "test"}, f)
        path = Path(f.name)
    with pytest.raises(ValueError, match="missing 'tiling_params'"):
        TilingSchema.from_json(path)


def test_from_json_type_defaults_to_int64():
    path = make_json([{"name": "X"}])  # no "type" key
    schema = TilingSchema.from_json(path)
    assert schema.fields[0]["type"] == "int64"


# ── pack ──────────────────────────────────────────────────────────────────────

def test_pack_happy_path_all_int64():
    schema = TilingSchema.from_json(FIXTURE)
    data = schema.pack(
        TB_M=16, TB_N=16,
        dim_arg0_0=64, dim_arg1_1=64,
        dim_arg0_1=64, dim_arg1_0=64,
    )
    # 6 × 8 bytes = 48
    assert len(data) == 48
    vals = struct.unpack_from("<6q", data)
    assert vals == (16, 16, 64, 64, 64, 64)


def test_pack_int32_vs_int64():
    path = make_json([
        {"name": "A", "type": "int64"},
        {"name": "B", "type": "int32"},
        {"name": "C", "type": "int64"},
    ])
    schema = TilingSchema.from_json(path)
    data = schema.pack(A=1, B=2, C=3)
    # int64(8) + int32(4) + int64(8) = 20 bytes
    assert len(data) == 20
    a = struct.unpack_from("<q", data, 0)[0]
    b = struct.unpack_from("<i", data, 8)[0]
    c = struct.unpack_from("<q", data, 12)[0]
    assert (a, b, c) == (1, 2, 3)


def test_pack_int32_alias():
    """int32_t is accepted as a type alias for int32."""
    path = make_json([{"name": "X", "type": "int32_t"}])
    schema = TilingSchema.from_json(path)
    data = schema.pack(X=42)
    assert len(data) == 4
    assert struct.unpack_from("<i", data)[0] == 42


def test_pack_missing_field():
    schema = TilingSchema.from_json(FIXTURE)
    with pytest.raises(ValueError) as exc:
        schema.pack(TB_M=16, TB_N=16)  # missing 4 fields
    msg = str(exc.value)
    assert "missing tiling fields" in msg
    assert "dim_arg0_0" in msg
    assert "expected" in msg


def test_pack_extra_field():
    schema = TilingSchema.from_json(FIXTURE)
    with pytest.raises(ValueError) as exc:
        schema.pack(
            TB_M=16, TB_N=16,
            dim_arg0_0=64, dim_arg1_1=64,
            dim_arg0_1=64, dim_arg1_0=64,
            EXTRA=99,
        )
    msg = str(exc.value)
    assert "unexpected tiling fields" in msg
    assert "EXTRA" in msg


def test_pack_field_order_in_output_matches_schema():
    """Output byte order follows schema declaration order, not kwargs order."""
    path = make_json([
        {"name": "FIRST", "type": "int64"},
        {"name": "SECOND", "type": "int64"},
    ])
    schema = TilingSchema.from_json(path)
    # Intentionally pass kwargs in reverse order
    data = schema.pack(SECOND=200, FIRST=100)
    first, second = struct.unpack_from("<2q", data)
    assert first == 100
    assert second == 200


def test_repr():
    schema = TilingSchema.from_json(FIXTURE)
    r = repr(schema)
    assert "TB_M" in r
    assert "TilingSchema" in r
