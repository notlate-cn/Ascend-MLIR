#!/usr/bin/env python3
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def assert_dag_tiling_default_preferred(path: Path) -> None:
    script = path.read_text(encoding="utf-8")
    if '"tilingParams": entry.get("tilingParams", {})' not in script:
        fail(f"{path}: per-kernel schema must preserve tilingParams")
    if '"scheduleEntries": entry.get("scheduleEntries", [])' not in script:
        fail(f"{path}: per-kernel schema must preserve scheduleEntries")
    if "schedule_defaults = {}" not in script:
        fail(f"{path}: run-mainline must read generated schedule tiling defaults")

    default_pos = script.find('elif "default" in field:')
    schedule_default_pos = script.find("elif name in schedule_defaults:")
    values_pos = script.find('elif field.get("values"):')
    if default_pos < 0:
        fail(f"{path}: tiling selection must prefer generated defaults")
    if schedule_default_pos < 0:
        fail(f"{path}: tiling selection must use schedule defaults")
    if values_pos < 0:
        fail(f"{path}: tiling selection must still support explicit values")
    if default_pos > schedule_default_pos or schedule_default_pos > values_pos:
        fail(f"{path}: tiling selection must check defaults before values")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        fail("usage: check_dag_tiling_defaults.py RUN_MAINLINE...")

    for raw_path in argv[1:]:
        assert_dag_tiling_default_preferred(Path(raw_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
