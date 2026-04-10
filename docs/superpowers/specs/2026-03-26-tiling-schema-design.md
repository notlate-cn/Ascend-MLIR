# TilingSchema Design

**Date:** 2026-03-26
**Scope:** Path B only — afir-opt pipeline → tiling_space.json → validator / Python hand-tuning
**Status:** Approved

---

## Problem

`RunArgs::tiling` is a raw `std::vector<uint8_t>`. Callers must know the field names, order,
and types (int32/int64) out-of-band, and pack bytes manually via `buildTiling()` with a string
like `"TB_M=16,TB_N=4,M=64,N=64"` plus a separate `--tiling-layout "int64,int64,int32,int32"`.

A mismatch (wrong field count, wrong order, wrong type) produces a silent kernel crash rather
than a meaningful error message.

The schema already exists: `afir-translate --tiling-space-out` emits `tiling_space.json`
with the `name` and `type` of every tiling field. This schema currently flows only to the
autotuner; it does not reach the validator or Python callers.

---

## Goals

- Validate tiling parameters (name, count, type) at pack time, not at kernel crash time.
- Expose a named-parameter API in both C++ and Python.
- Do not change `RunArgs::tiling`, the C API (`afirt_executor_run`), or `execute_kernel`.
- Do not add a new C API binding for TilingSchema; the Python layer is pure Python.

---

## Non-Goals

- Path A (AscGen / AutofuseTiling / 60-byte CANN struct) — out of scope.
- Dynamic schema loading at kernel launch time.
- Schema versioning or backward-compatibility checks.
- Modifying the tiling_space.json format.

---

## Design

### Layering

```
tiling_space.json
      │
      ▼
TilingSchema::fromJson()        ← C++: include/Runtime/TilingSchema.h
TilingSchema.from_json()        ← Python: python/runtime/tiling_schema.py
      │
      ▼  pack(params)   →  std::vector<uint8_t> / bytes
      │
      ▼
RunArgs::tiling                 ← unchanged
afirt_executor_run(tiling_data, tiling_len)  ← unchanged
execute_kernel(tiling_data=bytes)            ← unchanged
```

`TilingSchema` is a pure helper layer. It reads only `name` and `type` from
`tiling_params[]`; all autotuner-specific fields (`fixed`, `shape_key`, `min`, `max`,
`step`, `values`, `constraints`) are ignored.

---

### C++ API

```cpp
// include/Runtime/TilingSchema.h

struct TilingField {
  std::string name;
  std::string type;  // "int32" | "int64"
};

class TilingSchema {
public:
  /// Load schema from a tiling_space.json file.
  /// Returns error if the file cannot be read or is malformed.
  static llvm::Expected<TilingSchema> fromJson(llvm::StringRef path);

  /// Pack named parameters into a little-endian byte vector.
  /// params must be in field-declaration order; names are validated positionally.
  /// Returns error if count mismatches or a name does not match the schema field.
  llvm::Expected<std::vector<uint8_t>>
  pack(llvm::ArrayRef<std::pair<std::string, int64_t>> params) const;

  llvm::ArrayRef<TilingField> fields() const { return fields_; }
  size_t size() const { return fields_.size(); }

private:
  std::vector<TilingField> fields_;
};
```

**`pack()` contract:**
1. `params.size()` must equal `fields_.size()` — error otherwise.
2. For each position `i`: `params[i].first` must equal `fields_[i].name` — error otherwise.
3. Packing: `int32` → 4 bytes LE, `int64` → 8 bytes LE. Unknown type defaults to `int64`.

This matches the existing `buildTiling()` behavior exactly; `TilingSchema::pack()` is
a validated replacement for it.

**Implementation:** `lib/Runtime/TilingSchema.cpp` using `llvm::json::parse` (already a
project dependency via `LLVMSupport`).

---

### Python API

```python
# python/runtime/tiling_schema.py

class TilingSchema:
    @classmethod
    def from_json(cls, path: str | Path) -> "TilingSchema":
        """Load schema from tiling_space.json. Reads only name and type fields."""
        ...

    def pack(self, **kwargs: int) -> bytes:
        """
        Pack named tiling parameters into little-endian bytes.

        Parameters must cover every schema field exactly (no extras, no missing).
        Field order in the output follows the schema declaration order.

        Raises:
            ValueError: missing fields, extra fields, or unrecognised field name.
                        The message lists expected fields for easy debugging.
        """
        ...

    @property
    def field_names(self) -> list[str]:
        """Ordered list of field names as declared in the schema."""
        ...

    @property
    def fields(self) -> list[dict]:
        """Ordered list of {name, type} dicts."""
        ...
```

**`pack()` error messages (examples):**

```
ValueError: missing tiling fields: ['TB_N', 'M']
  expected: ['TB_M', 'TB_N', 'M', 'N']

ValueError: unexpected tiling fields: ['TB_X']
  expected: ['TB_M', 'TB_N', 'M', 'N']
```

`from_json` is pure Python (`json.load`), no C extension required.

---

### validator CLI change

Add `--tiling-schema <path>` as an alternative to `--tiling-layout <types>`:

```bash
# Before (still works):
validator --tiling-params "TB_M=16,TB_N=4,M=64,N=64" \
          --tiling-layout "int64,int64,int64,int64" ...

# After (preferred):
validator --tiling-params "TB_M=16,TB_N=4,M=64,N=64" \
          --tiling-schema step8.tiling_space.json ...
```

When `--tiling-schema` is given, the validator loads the schema, calls `TilingSchema::pack()`,
and emits a clear error if params mismatch. `--tiling-layout` remains supported; if both are
provided, `--tiling-schema` takes precedence.

---

## File Changelist

| Change | File | Notes |
|--------|------|-------|
| Add    | `include/Runtime/TilingSchema.h` | C++ header — TilingField + TilingSchema |
| Add    | `lib/Runtime/TilingSchema.cpp` | Implementation using llvm::json |
| Modify | `lib/Runtime/CMakeLists.txt` | Add TilingSchema.cpp to sources |
| Add    | `python/runtime/tiling_schema.py` | Pure-Python TilingSchema |
| Modify | `python/runtime/__init__.py` | Export TilingSchema |
| Modify | `tools/validator/validator_main.cpp` | Add --tiling-schema option |
| Add    | `test/tools/runtime/test_tiling_schema.cpp` | C++ unit tests |
| Add    | `python/test/test_tiling_schema.py` | Python unit tests |

---

## Test Plan

### C++ unit tests (`test_tiling_schema.cpp`)

- `fromJson` roundtrip: load a JSON fixture, verify `fields()` names and types.
- `pack` happy path: correct params produce expected bytes (verified against manual `buildTiling` output).
- `pack` error — wrong count: returns error with informative message.
- `pack` error — wrong name at position: returns error with informative message.
- `fromJson` error — file not found: returns error.
- `fromJson` error — malformed JSON: returns error.

### Python unit tests (`test_tiling_schema.py`)

- `from_json` roundtrip: load fixture, verify `field_names`.
- `pack` happy path: output bytes match reference (struct.pack equivalent).
- `pack` missing field: raises `ValueError` with message listing missing fields.
- `pack` extra field: raises `ValueError` with message listing unexpected fields.
- `pack` int32 vs int64: verify 4-byte vs 8-byte packing.

### Integration

Existing `broadcast-add-reduce` example's `run.sh` validator invocation updated to use
`--tiling-schema` instead of `--tiling-layout`. All existing tests must continue to pass.

---

## Migration Path

1. Ship `TilingSchema` alongside existing `buildTiling` — no breaking change.
2. Update `broadcast-add-reduce/run.sh` validator call to use `--tiling-schema`.
3. Autotuner can optionally migrate to `TilingSchema::pack()` in a follow-up (it already
   has the parsed schema; the migration is mechanical).
4. `buildTiling()` and `--tiling-layout` remain; no forced migration.
