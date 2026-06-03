from __future__ import annotations

import pathlib
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass


class CommandError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        argv: tuple[str, ...] | None = None,
        returncode: int | None = None,
        stderr: str | None = None,
        stdout_path: pathlib.Path | None = None,
        stderr_report_path: pathlib.Path | None = None,
    ) -> None:
        super().__init__(message)
        self.argv = argv
        self.returncode = returncode
        self.stderr = stderr
        self.stdout_path = stdout_path
        self.stderr_report_path = stderr_report_path


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stderr: str


def find_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    raise CommandError(f"required tool not found in PATH: {name}")


def _write_report(path: pathlib.Path, *, argv: list[str], returncode: int, stderr: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        f"command: {shlex.join(argv)}\n"
        f"exit_code: {returncode}\n"
        "stderr:\n"
        f"{stderr}"
    )
    path.write_text(body, encoding="utf-8")


def run_command(
    argv: list[str],
    *,
    stdout_path: pathlib.Path | None = None,
    stderr_report_path: pathlib.Path | None = None,
) -> CommandResult:
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

    if stderr_report_path:
        _write_report(
            stderr_report_path,
            argv=argv,
            returncode=completed.returncode,
            stderr=completed.stderr,
        )

    if completed.returncode != 0:
        if temp_path:
            temp_path.unlink(missing_ok=True)
        command = shlex.join(argv)
        raise CommandError(
            f"command failed with exit code {completed.returncode}: {command}\n{completed.stderr}",
            argv=tuple(argv),
            returncode=completed.returncode,
            stderr=completed.stderr,
            stdout_path=stdout_path,
            stderr_report_path=stderr_report_path,
        )

    if temp_path:
        temp_path.replace(stdout_path)
    return CommandResult(tuple(argv), completed.returncode, completed.stderr)
