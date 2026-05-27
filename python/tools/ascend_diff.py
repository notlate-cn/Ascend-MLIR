#!/usr/bin/env python3
"""ascend-diff: compare actual vs expected .npy tensors with PASS/FAIL judgement.

P0-2 of docs/auto-fuse/debug.md §7.6 (L0 final-output diff). Collects the
ad-hoc np.allclose snippets scattered in /tmp/*.py into a single tool with
consistent atol/rtol semantics and a clean CI exit code.

Usage:
  ascend-diff --actual a.npy b.npy --expected x.npy y.npy
              [--atol 1e-3] [--rtol 1e-2]
              [--names net.out[0] net.out[1]]
              [--json]

Exit codes: 0 = all PASS; 1 = any FAIL; 2 = argument/IO error.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def diff_pair(actual_path: Path, expected_path: Path,
              atol: float, rtol: float) -> dict:
    """Compare one (actual, expected) npy pair; return a result dict."""
    try:
        a = np.load(actual_path).astype(np.float32)
        b = np.load(expected_path).astype(np.float32)
    except Exception as e:
        return {"ok": False, "kind": "io_error", "error": str(e),
                "actual": str(actual_path), "expected": str(expected_path)}

    result = {
        "actual": str(actual_path),
        "expected": str(expected_path),
        "shape": list(a.shape),
        "expected_shape": list(b.shape),
        "atol": atol,
        "rtol": rtol,
    }
    if a.shape != b.shape:
        result.update({"ok": False, "kind": "shape_mismatch"})
        return result
    if a.size == 0:
        result.update({"ok": True, "kind": "empty",
                       "max_diff": 0.0, "mean_diff": 0.0,
                       "first_mismatch": None})
        return result

    diff = np.abs(a - b)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    ok = bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False))

    # Mirror np.allclose's element-wise predicate to find the first offender.
    first_mismatch = None
    if not ok:
        mask = diff > (atol + rtol * np.abs(b))
        idx = np.argwhere(mask)
        if idx.size > 0:
            first_mismatch = [int(x) for x in idx[0]]

    result.update({
        "ok": ok,
        "kind": "ok" if ok else "out_of_tolerance",
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "first_mismatch": first_mismatch,
    })
    return result


def render_text(results, names, summary):
    for name, r in zip(names, results):
        if r["kind"] == "io_error":
            print(f"{name}: IO_ERROR {r['error']}  FAIL")
        elif r["kind"] == "shape_mismatch":
            print(f"{name}: SHAPE_MISMATCH "
                  f"got={tuple(r['shape'])} want={tuple(r['expected_shape'])}  FAIL")
        else:
            tag = "PASS" if r["ok"] else f"FAIL (atol={r['atol']:.1e} rtol={r['rtol']:.1e})"
            fm = (f" first_mismatch={r['first_mismatch']}"
                  if r["first_mismatch"] is not None else "")
            print(f"{name}: shape={tuple(r['shape'])} "
                  f"max_diff={r['max_diff']:.4g} mean_diff={r['mean_diff']:.4g}{fm}  {tag}")
    print(f"{summary['pass']}/{summary['pass'] + summary['fail']} PASS")


def main():
    ap = argparse.ArgumentParser(
        prog="ascend-diff",
        description="Compare actual vs expected .npy tensors with PASS/FAIL.")
    ap.add_argument("--actual", nargs="+", required=True,
                    help="Actual output .npy paths.")
    ap.add_argument("--expected", nargs="+", required=True,
                    help="Expected reference .npy paths (must match --actual count).")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--names", nargs="+", default=None,
                    help="Optional human-readable label per pair; defaults to out[i].")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON to stdout instead of human-readable text.")
    args = ap.parse_args()

    if len(args.actual) != len(args.expected):
        print(f"error: --actual count ({len(args.actual)}) != "
              f"--expected count ({len(args.expected)})", file=sys.stderr)
        sys.exit(2)
    if args.names and len(args.names) != len(args.actual):
        print(f"error: --names count ({len(args.names)}) != "
              f"--actual count ({len(args.actual)})", file=sys.stderr)
        sys.exit(2)

    names = args.names or [f"out[{i}]" for i in range(len(args.actual))]
    results = [
        diff_pair(Path(a), Path(e), args.atol, args.rtol)
        for a, e in zip(args.actual, args.expected)
    ]
    summary = {
        "pass": sum(1 for r in results if r.get("ok")),
        "fail": sum(1 for r in results if not r.get("ok")),
    }

    if args.json:
        json.dump({"results": [{"name": n, **r} for n, r in zip(names, results)],
                   "summary": summary},
                  sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        render_text(results, names, summary)

    sys.exit(0 if summary["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
