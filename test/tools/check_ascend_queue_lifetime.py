#!/usr/bin/env python3
"""Check that AscendC dequeued tensors are released exactly once."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VALUE = r"%[A-Za-z0-9_.$#-]+"
DEQUE_RE = re.compile(
    rf"(?P<value>{VALUE})\s*=\s*ascendc(?:\.que_bind)?\.deque(?:_tensor)?\s+"
    rf"(?P<queue>{VALUE})\b"
)
FREE_RE = re.compile(
    rf"ascendc(?:\.que_bind)?\.free_tensor\s+(?P<queue>{VALUE})\s*,\s*"
    rf"(?P<value>{VALUE})\b"
)
FUNC_RE = re.compile(r"\bfunc\.func\b")


@dataclass
class DequeRecord:
    value: str
    queue: str
    line: int
    free_count: int = 0
    duplicate_line: int | None = None


@dataclass
class FileStats:
    deque_count: int = 0
    free_count: int = 0


def iter_files(paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.mlir") if p.is_file()))
        else:
            files.append(path)
    return files


def flush_scope(path: Path, records: list[DequeRecord], errors: list[str]) -> None:
    for record in records:
        if record.free_count == 0:
            errors.append(
                f"{path}:{record.line}: missing free_tensor for deque result "
                f"{record.value} from queue {record.queue}"
            )
        elif record.free_count > 1:
            line = record.duplicate_line or record.line
            errors.append(
                f"{path}:{line}: duplicate free_tensor for deque result "
                f"{record.value} from queue {record.queue}"
            )


def analyze_file(path: Path) -> tuple[FileStats, list[str]]:
    stats = FileStats()
    errors: list[str] = []
    records: list[DequeRecord] = []

    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if FUNC_RE.search(line):
                flush_scope(path, records, errors)
                records = []

            deque_match = DEQUE_RE.search(line)
            if deque_match:
                records.append(
                    DequeRecord(
                        value=deque_match.group("value"),
                        queue=deque_match.group("queue"),
                        line=line_no,
                    )
                )
                stats.deque_count += 1
                continue

            free_match = FREE_RE.search(line)
            if not free_match:
                continue
            stats.free_count += 1

            value = free_match.group("value")
            queue = free_match.group("queue")
            for record in reversed(records):
                if record.value == value and record.queue == queue:
                    record.free_count += 1
                    if record.free_count == 2:
                        record.duplicate_line = line_no
                    break

    flush_scope(path, records, errors)
    return stats, errors


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="MLIR file or directory to check")
    args = parser.parse_args(argv)

    all_errors: list[str] = []
    total_deque = 0
    total_free = 0
    checked = 0

    for path in iter_files(args.paths):
        if not path.exists():
            print(f"{path}: no such file", file=sys.stderr)
            return 2

        stats, errors = analyze_file(path)
        total_deque += stats.deque_count
        total_free += stats.free_count
        checked += 1
        all_errors.extend(errors)

    if all_errors:
        for error in all_errors:
            print(error, file=sys.stderr)
        return 1

    label = os.path.basename(args.paths[0])
    if checked != 1:
        label = f"{checked} files"
    print(f"queue_lifetime.ok {label} deque={total_deque} free={total_free}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
