# python/runtime/tiling_schema.py
"""
TilingSchema: load a tiling_space.json and pack named tiling parameters
into little-endian bytes suitable for execute_kernel(tiling_data=...).

Only 'name' and 'type' fields in tiling_params[] are used.
Autotuner fields (fixed, shape_key, min, max, step, values) are ignored.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Union


class TilingSchema:
    """
    Validated tiling parameter packer loaded from a tiling_space.json file.

    Example::

        schema = TilingSchema.from_json("step8.tiling_space.json")
        tiling_bytes = schema.pack(TB_M=16, TB_N=4, M=64, N=64)
        execute_kernel(..., tiling_data=tiling_bytes)
    """

    def __init__(self, fields: list[dict]) -> None:
        # fields: list of {"name": str, "type": "int32"|"int64"}
        self._fields = fields

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TilingSchema":
        """
        Load schema from a tiling_space.json file.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: if the JSON is malformed or missing 'tiling_params'.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"tiling_space.json not found: {path}")
        with path.open() as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error in '{path}': {e}") from e
        params = data.get("tiling_params")
        if params is None:
            raise ValueError(f"'{path}': missing 'tiling_params' array")
        if not isinstance(params, list):
            raise ValueError(f"'{path}': 'tiling_params' must be a list")
        fields = []
        for i, entry in enumerate(params):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"'{path}': tiling_params[{i}] must be an object"
                )
            name = entry.get("name")
            if not name:
                raise ValueError(
                    f"'{path}': tiling_params[{i}] missing 'name'"
                )
            fields.append({
                "name": name,
                "type": entry.get("type", "int64"),
            })
        return cls(fields)

    def pack(self, **kwargs: int) -> bytes:
        """
        Pack named tiling parameters into little-endian bytes.

        Parameters must cover every schema field exactly (no extras, no missing).
        Field order in the output follows the schema declaration order.

        Args:
            **kwargs: one keyword argument per schema field, e.g.
                      schema.pack(TB_M=16, TB_N=4, M=64, N=64)

        Returns:
            bytes: packed little-endian representation

        Raises:
            ValueError: if fields are missing, extra, or names do not match.
        """
        expected = [f["name"] for f in self._fields]
        given = set(kwargs.keys())
        expected_set = set(expected)

        missing = sorted(expected_set - given)
        extra = sorted(given - expected_set)

        errors = []
        if missing:
            errors.append(f"missing tiling fields: {missing}")
        if extra:
            errors.append(f"unexpected tiling fields: {extra}")
        if errors:
            raise ValueError(
                "\n".join(errors) + f"\n  expected: {expected}"
            )

        result = bytearray()
        for field in self._fields:
            val = int(kwargs[field["name"]])
            if field["type"] in ("int32", "int32_t"):
                result += struct.pack("<i", val)
            else:
                result += struct.pack("<q", val)
        return bytes(result)

    @property
    def field_names(self) -> list[str]:
        """Ordered list of field names as declared in the schema."""
        return [f["name"] for f in self._fields]

    @property
    def fields(self) -> list[dict]:
        """Ordered list of {'name': str, 'type': str} dicts."""
        return list(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __repr__(self) -> str:
        return f"TilingSchema({self.field_names})"
