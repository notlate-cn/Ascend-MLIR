from __future__ import annotations

import pathlib
import shutil
import subprocess


class CommandError(RuntimeError):
    pass


def find_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    raise CommandError(f"required tool not found in PATH: {name}")


def run_command(argv: list[str], *, stdout_path: pathlib.Path | None = None) -> None:
    if stdout_path:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_file = stdout_path.open("w", encoding="utf-8")
    else:
        stdout_file = subprocess.DEVNULL

    try:
        completed = subprocess.run(
            argv,
            stdout=stdout_file,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    finally:
        if stdout_path:
            stdout_file.close()

    if completed.returncode != 0:
        command = " ".join(argv)
        raise CommandError(
            f"command failed with exit code {completed.returncode}: {command}\n{completed.stderr}"
        )
