from __future__ import annotations

import pathlib
import shlex
import shutil
import subprocess
import tempfile


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
        stdout_file = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=stdout_path.parent,
            prefix=f".{stdout_path.name}.",
            suffix=".tmp",
        )
        temp_path = pathlib.Path(stdout_file.name)
    else:
        stdout_file = subprocess.DEVNULL
        temp_path = None

    run_failed = False
    try:
        completed = subprocess.run(
            argv,
            stdout=stdout_file,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except BaseException:
        run_failed = True
        raise
    finally:
        if stdout_path:
            stdout_file.close()
        if run_failed and temp_path:
            temp_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        if temp_path:
            temp_path.unlink(missing_ok=True)
        command = shlex.join(argv)
        raise CommandError(
            f"command failed with exit code {completed.returncode}: {command}\n{completed.stderr}"
        )

    if temp_path:
        temp_path.replace(stdout_path)
