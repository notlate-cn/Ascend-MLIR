#!/usr/bin/env python3
from __future__ import annotations

import ast
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_lit_suite_name.py <lit.cfg.py>", file=sys.stderr)
        return 2

    path = pathlib.Path(sys.argv[1])
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "config"
                and target.attr == "name"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                print(f"lit.suite.name={node.value.value}")
                return 0

    print("lit.suite.name=<missing>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
