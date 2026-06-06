from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class ContractBundle:
    root: pathlib.Path
    contracts: dict[str, dict[str, Any]]

    def has(self, schema: str) -> bool:
        return schema in self.contracts

    def get(self, schema: str) -> dict[str, Any]:
        return self.contracts[schema]

    def available_schemas(self) -> list[str]:
        return sorted(self.contracts)


def load_contract_file(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid debug contract {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"debug contract must be an object: {path}")
    schema = value.get("schema")
    version = value.get("schema_version")
    data = value.get("data")
    if not isinstance(schema, str) or not schema.startswith("ascend.debug."):
        raise ContractError(f"debug contract missing valid schema: {path}")
    if version != 1:
        raise ContractError(
            f"unsupported debug contract version in {path}: {version}"
        )
    if not isinstance(data, dict):
        raise ContractError(f"debug contract data must be an object: {path}")
    return value


def load_contract_bundle(root: pathlib.Path) -> ContractBundle:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise ContractError(f"debug contract directory not found: {resolved}")
    contracts: dict[str, dict[str, Any]] = {}
    for path in sorted(resolved.glob("*.json")):
        contract = load_contract_file(path)
        contracts[contract["schema"]] = contract
    return ContractBundle(root=resolved, contracts=contracts)
