from __future__ import annotations

import pathlib
from typing import Any

from ascend_debug import layout
from ascend_debug.runner import CommandError


STATUS_REL_PATH = "run_status.json"


def command_record(
    *,
    stage: str,
    tool: str,
    args: list[str],
    stdout: str | None = None,
    stderr: str | None = None,
    status: str = "success",
    exit_code: int | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stage": stage,
        "tool": tool,
        "args": args,
        "status": status,
    }
    if stdout:
        record["stdout"] = stdout
    if stderr:
        record["stderr"] = stderr
    if exit_code is not None:
        record["exit_code"] = exit_code
    if message:
        record["message"] = message
    return record


def command_error_message(error: BaseException) -> str:
    stderr = getattr(error, "stderr", None)
    if isinstance(stderr, str) and stderr.strip():
        return stderr.strip()
    return str(error).strip()


def failed_command_record(
    *,
    stage: str,
    tool: str,
    args: list[str],
    stdout: str | None,
    stderr: str | None,
    error: CommandError,
) -> dict[str, Any]:
    return command_record(
        stage=stage,
        tool=tool,
        args=args,
        stdout=stdout,
        stderr=stderr,
        status="failed",
        exit_code=error.returncode,
        message=command_error_message(error),
    )


def command_from_error(error: BaseException) -> dict[str, Any] | None:
    command = getattr(error, "debug_command", None)
    return command if isinstance(command, dict) else None


def report_records_from_commands(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    seen = set()
    for command in commands:
        stage = command.get("stage")
        stderr = command.get("stderr")
        if not isinstance(stderr, str) or not stderr or stderr in seen:
            continue
        reports.append({"stage": stage if isinstance(stage, str) else "", "path": stderr})
        seen.add(stderr)
    return reports


def existing_stages(
    run_dir: pathlib.Path,
    stages: tuple[layout.StageArtifact, ...],
) -> tuple[layout.StageArtifact, ...]:
    return tuple(stage for stage in stages if (run_dir / stage.path).exists())


def write_run_status(
    run_dir: pathlib.Path,
    *,
    stage: str,
    phase: str,
    command: dict[str, Any] | None,
    error: BaseException,
) -> str:
    document = {
        "schema_version": 1,
        "tool": "ascend-debug",
        "status": "failed",
        "failure": {
            "stage": stage,
            "phase": phase,
            "message": command_error_message(error),
            "command": command or {},
        },
    }
    layout.write_json(run_dir / STATUS_REL_PATH, document)
    return STATUS_REL_PATH
