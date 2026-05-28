"""Subprocess wrapper: echoes the command and (when logger is initialized) tees
the child's stderr into ``<log_dir>/<phase>.stderr`` while preserving real-time
output on the parent's stderr.

Falls back to a plain ``subprocess.run`` echo when the logger hasn't been
initialized, so import-only consumers and ad-hoc scripts keep working.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from typing import Any

from . import logger as _lg


def _tee(src, dst_file, dst_buf) -> None:
    try:
        for chunk in iter(lambda: src.readline(), b""):
            if not chunk:
                break
            dst_file.write(chunk)
            dst_file.flush()
            dst_buf.write(chunk)
            dst_buf.flush()
    finally:
        try:
            src.close()
        except Exception:
            pass


def run(cmd, **kw: Any):
    """Run ``cmd`` with the usual ``check=True`` default; tee stderr if possible."""
    cmd_str = " ".join(str(c) for c in cmd)
    if _lg.is_initialized():
        _lg.passthrough("+ " + cmd_str)
    else:
        print("+", cmd_str, flush=True)

    check = kw.pop("check", True)
    log_dir = _lg.get_log_dir() if _lg.is_initialized() else None
    user_stderr = kw.get("stderr") is not None
    capture_output = kw.get("capture_output")

    if log_dir is None or user_stderr or capture_output:
        # Caller wants to control stderr (or logger not ready) -> straight passthrough.
        return subprocess.run(cmd, check=check, **kw)

    phase = _lg.current_phase()
    stderr_path = log_dir / f"{phase}.stderr"
    fh = open(stderr_path, "ab")
    fh.write(f"\n+ {cmd_str}\n".encode("utf-8"))
    fh.flush()

    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, **kw)
    t = threading.Thread(target=_tee, args=(proc.stderr, fh, sys.stderr.buffer))
    t.start()
    rc = proc.wait()
    t.join()
    fh.close()

    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return subprocess.CompletedProcess(cmd, rc)
