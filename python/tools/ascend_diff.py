#!/usr/bin/env python3
"""ascend-diff: PASS/FAIL diff for two tensor sets, plus first-bad-kernel
topological scan when given parallel per-kernel checkpoint dirs.

Modes
-----
- L0 final-output diff (P0-2, default — backward-compatible CLI):
    ascend-diff --actual a.npy [b.npy ...] --expected x.npy [y.npy ...]
                [--atol 1e-3] [--rtol 1e-2]
                [--names net.out[0] ...] [--json]

- L1 first-bad-kernel locate (P2 Alt 2 — sim-as-reference path):
    ascend-diff locate --network <network.json>
                       --actual <dir-with-<kernel>_out_<i>.npy>
                       --reference <dir-with-<kernel>_out_<i>.npy>
                       [--provenance <network.provenance.json>]
                       [--profiles <profiles-dir>]
                       [--atol] [--rtol] [--json]

  Walks the AscendC kernels in network.json topological order; for each
  per-kernel output, compares actual vs reference (sim/cpu/whatever); the
  first kernel that diverges while all its predecessors PASSed is the
  first-bad-kernel. aclnn kernels are skipped (HostLaunchHelper only dumps
  intermediates for AscendC kernels).

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


def main_diff(argv):
    ap = argparse.ArgumentParser(
        prog="ascend-diff",
        description="L0 final-output diff: actual vs expected .npy tensors.")
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
    args = ap.parse_args(argv)

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
        json.dump({"mode": "diff",
                   "results": [{"name": n, **r} for n, r in zip(names, results)],
                   "summary": summary},
                  sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        render_text(results, names, summary)

    sys.exit(0 if summary["fail"] == 0 else 1)


def _build_predecessors(kernels):
    """Build kernel_id → list of predecessor kernel_ids from network.json."""
    preds = {}
    for k in kernels:
        kid = k["id"]
        ps = []
        for arg in k.get("args", []):
            if arg.get("from") == "kernel":
                src = arg["kernel"]
                if src not in ps:
                    ps.append(src)
        preds[kid] = ps
    return preds


def main_locate(argv):
    ap = argparse.ArgumentParser(
        prog="ascend-diff locate",
        description="L1 first-bad-kernel topological scan (sim-as-reference).")
    ap.add_argument("--network", required=True,
                    help="Path to network.json (topology).")
    ap.add_argument("--actual", required=True,
                    help="Dir containing <kernel>_out_<i>.npy from the suspect run.")
    ap.add_argument("--reference", required=True,
                    help="Dir containing <kernel>_out_<i>.npy from the trusted run.")
    ap.add_argument("--provenance", default=None,
                    help="Optional network.provenance.json (enriches the report).")
    ap.add_argument("--profiles", default=None,
                    help="Optional profiles/ dir (kernel timing summary in report).")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    network = json.loads(Path(args.network).read_text())
    prov_by_id = {}
    if args.provenance:
        prov = json.loads(Path(args.provenance).read_text())
        prov_by_id = {k["kernel_id"]: k for k in prov.get("kernels", [])}
    actual_dir = Path(args.actual)
    ref_dir = Path(args.reference)
    profiles_dir = Path(args.profiles) if args.profiles else None

    kernels = network.get("kernels", [])
    preds = _build_predecessors(kernels)

    # Locate intermediate npys. HostLaunchHelper dumps per-VARIANT name
    # (kernel_group0__v0), but network.json's kernel id is kernel_group0.
    # Probe a small set of likely suffixes.
    def find_npy(d: Path, kid: str, oidx: int):
        for suffix in ("", "__v0", "__v1", "__v2"):
            p = d / f"{kid}{suffix}_out_{oidx}.npy"
            if p.exists():
                return p
        return None

    per_kernel = []
    pass_set = set()
    first_bad = None

    for k in kernels:
        kid = k["id"]
        if k.get("kind") != "ascendc":
            per_kernel.append({"kernel_id": kid, "kind": k.get("kind"),
                               "status": "skipped",
                               "reason": "non-ascendc (no per-kernel dump)"})
            pass_set.add(kid)  # not a failure source either way
            continue

        n_results = len(k.get("results", []))
        outputs = []
        all_ok = True
        any_io_err = False
        for i in range(n_results):
            a_path = find_npy(actual_dir, kid, i)
            r_path = find_npy(ref_dir, kid, i)
            if a_path is None or r_path is None:
                outputs.append({"result_index": i, "status": "missing",
                                "actual": str(a_path) if a_path else None,
                                "reference": str(r_path) if r_path else None})
                all_ok = False
                any_io_err = True
                continue
            r = diff_pair(a_path, r_path, args.atol, args.rtol)
            outputs.append({"result_index": i, **r})
            if not r.get("ok"):
                all_ok = False

        # Optional timing from profiles/<kid>(__vN)?.timing.json
        timing = None
        if profiles_dir:
            for suffix in ("", "__v0", "__v1"):
                p = profiles_dir / f"{kid}{suffix}.timing.json"
                if p.exists():
                    try:
                        timing = json.loads(p.read_text())
                    except Exception:
                        pass
                    break

        prov = prov_by_id.get(kid, {})
        record = {
            "kernel_id": kid,
            "kind": "ascendc",
            "fused_ops_summary": prov.get("fused_ops_summary"),
            "source_ops": [{"id": s.get("id"),
                            "op_role": s.get("op_role"),
                            "loc": s.get("loc")}
                           for s in prov.get("source_ops", [])],
            "outputs": outputs,
            "status": "PASS" if all_ok else ("MISSING" if any_io_err else "FAIL"),
            "timing": timing,
        }
        per_kernel.append(record)

        if all_ok:
            pass_set.add(kid)
        elif first_bad is None and all(p in pass_set for p in preds.get(kid, [])):
            first_bad = kid

    summary = {
        "total": len(kernels),
        "ascendc_scanned": sum(1 for r in per_kernel if r.get("kind") == "ascendc"),
        "pass": sum(1 for r in per_kernel if r.get("status") == "PASS"),
        "fail": sum(1 for r in per_kernel if r.get("status") == "FAIL"),
        "missing": sum(1 for r in per_kernel if r.get("status") == "MISSING"),
        "first_bad_kernel": first_bad,
    }

    if args.json:
        json.dump({"mode": "locate", "kernels": per_kernel, "summary": summary},
                  sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for r in per_kernel:
            if r["status"] == "skipped":
                continue
            kid = r["kernel_id"]
            mark = "[FIRST-BAD]" if kid == first_bad else ""
            tag = r["status"]
            outs_summary = ", ".join(
                ("missing" if o.get("status") == "missing"
                 else f"max_diff={o.get('max_diff', 0):.4g}")
                for o in r["outputs"])
            summ = r.get("fused_ops_summary") or "?"
            tline = ""
            if r.get("timing") and r["timing"].get("wall_us") is not None:
                tline = f"  [{r['timing']['wall_us']}us]"
            print(f"{kid} [{summ}] {tag}{tline}  ({outs_summary})  {mark}")
        print(f"---")
        print(f"scanned {summary['ascendc_scanned']} AscendC kernels: "
              f"{summary['pass']} PASS / {summary['fail']} FAIL / "
              f"{summary['missing']} MISSING")
        if first_bad:
            print(f"first-bad-kernel: {first_bad}")
            fb = next((r for r in per_kernel if r["kernel_id"] == first_bad), None)
            if fb:
                for so in fb.get("source_ops", []):
                    print(f"  source op {so['id']}: {so['op_role']}  {so.get('loc','')}")
        else:
            print("first-bad-kernel: none (all AscendC kernels PASS)")

    sys.exit(0 if summary["fail"] == 0 and summary["missing"] == 0 else 1)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "locate":
        return main_locate(argv[1:])
    return main_diff(argv)


if __name__ == "__main__":
    main()
