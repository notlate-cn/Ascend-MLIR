"""Unified Python-side debug logger for network_runner and friends.

Sinks: stderr (INFO+, color when TTY) and <workdir>/log/run.log (DEBUG+).
The run.log file APPENDS across re-runs (each session prefixed by a
banner line); use the `run-id` prefix on every log line to grep a single
session out.

Context: contextvars-driven phase/kernel prefix, grep-friendly.  Prefer
the `context()` contextmanager when entering a scoped phase — it auto-
restores the prior phase/kernel on exit, so cleanup code after the
`with` block doesn't carry the inner phase's tag.

Typical usage:
    from runner_utils.logger import init_run, get_logger, context, passthrough
    init_run(workdir, run_id=args.run_id, verbose=args.verbose)
    log = get_logger(__name__)
    with context(phase="phase-2"):
        log.info("matmul -> aclnn picked v0")
        passthrough("+ afir-opt ...")   # raw line, no prefix
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

_PASSTHROUGH = 25
logging.addLevelName(_PASSTHROUGH, "PASS")

_ctx_phase: contextvars.ContextVar[str] = contextvars.ContextVar("phase", default="-")
_ctx_kernel: contextvars.ContextVar[str] = contextvars.ContextVar("kernel", default="-")

_state = {
    "workdir": None,
    "run_id": None,
    "log_dir": None,
    "verbose": False,
    "initialized": False,
}


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _state.get("run_id") or "-"
        record.phase = _ctx_phase.get()
        record.kernel = _ctx_kernel.get()
        return True


class _Formatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[2m",
        "INFO": "",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[31;1m",
    }
    RESET = "\033[0m"

    def __init__(self, *, color: bool):
        super().__init__()
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == _PASSTHROUGH:
            return record.getMessage()
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        lvl_short = {"WARNING": "WARN", "CRITICAL": "CRIT"}.get(
            record.levelname, record.levelname)
        lvl = lvl_short[:5].ljust(5)
        prefix = f"{ts} {lvl} [{record.run_id}][{record.phase}][{record.kernel}]"
        msg = f"{prefix} {record.getMessage()}"
        if self.color:
            c = self.COLORS.get(record.levelname, "")
            if c:
                return f"{c}{msg}{self.RESET}"
        return msg


def _default_run_id(workdir: Path) -> str:
    # PID disambiguates same-second launches (CI parallel, quick re-runs).
    base = workdir.name or "run"
    return f"{base}-{time.strftime('%H%M%S')}-{os.getpid()}"


def init_run(workdir, run_id: Optional[str] = None, verbose: bool = False) -> str:
    """Initialize sinks. Idempotent: removes prior handlers, reinstalls fresh."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rid = run_id or _default_run_id(workdir)
    log_dir = workdir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "run-id").write_text(rid + "\n")

    _state["workdir"] = workdir
    _state["run_id"] = rid
    _state["log_dir"] = log_dir
    _state["verbose"] = verbose

    root = logging.getLogger("ascend")
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.propagate = False

    cf = _ContextFilter()

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(_Formatter(color=sys.stderr.isatty()))
    sh.addFilter(cf)
    root.addHandler(sh)

    # Append, don't truncate: a re-run on the same workdir would otherwise
    # delete the prior session's log. The per-line run_id prefix makes a
    # single session greppable; the banner below makes session boundaries
    # eyeball-able in `tail -f` mode.
    log_path = log_dir / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as banner_fh:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        banner_fh.write(f"\n=== run {rid} start {ts} ===\n")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_Formatter(color=False))
    fh.addFilter(cf)
    root.addHandler(fh)

    _state["initialized"] = True
    return rid


def _ensure_stderr_only() -> None:
    """Attach a stderr-only handler so standalone tools (no workdir) still log."""
    root = logging.getLogger("ascend")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    root.propagate = False
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(_Formatter(color=sys.stderr.isatty()))
    sh.addFilter(_ContextFilter())
    root.addHandler(sh)


def get_logger(name: str = "ascend") -> logging.Logger:
    """Return a logger under the 'ascend' namespace.

    If ``init_run`` was never called (standalone tool use), a stderr-only
    handler is attached lazily so log calls aren't silently dropped.
    """
    _ensure_stderr_only()
    if not name or name == "ascend":
        return logging.getLogger("ascend")
    if name == "__main__":
        return logging.getLogger("ascend.main")
    if name.startswith("ascend."):
        return logging.getLogger(name)
    return logging.getLogger(f"ascend.{name}")


def set_context(*, phase: Optional[str] = None, kernel: Optional[str] = None) -> None:
    """Imperative phase/kernel set. Prefer `context()` for scoped use."""
    if phase is not None:
        _ctx_phase.set(phase or "-")
    if kernel is not None:
        _ctx_kernel.set(kernel or "-")


def clear_context() -> None:
    _ctx_phase.set("-")
    _ctx_kernel.set("-")


@contextlib.contextmanager
def context(*, phase: Optional[str] = None,
            kernel: Optional[str] = None) -> Iterator[None]:
    """Scoped phase/kernel tag; restores the prior values on exit.

    Use around a logical phase block so cleanup code after the `with`
    doesn't carry an inner phase tag:

        with context(phase="phase-3"):
            run_phase_3()
        # log lines after this point revert to the outer phase
    """
    prev_phase = _ctx_phase.get()
    prev_kernel = _ctx_kernel.get()
    if phase is not None:
        _ctx_phase.set(phase or "-")
    if kernel is not None:
        _ctx_kernel.set(kernel or "-")
    try:
        yield
    finally:
        _ctx_phase.set(prev_phase)
        _ctx_kernel.set(prev_kernel)


def current_phase() -> str:
    return _ctx_phase.get()


def get_log_dir() -> Optional[Path]:
    return _state.get("log_dir")


def is_initialized() -> bool:
    return bool(_state.get("initialized"))


def passthrough(msg: str) -> None:
    """Raw line (no prefix). Use for command echo and similar verbatim output."""
    logging.getLogger("ascend").log(_PASSTHROUGH, msg)
