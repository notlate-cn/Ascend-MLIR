# ascend-debug Phase 2a — L0 Tensor Diff Panel (sim)

**Date:** 2026-06-04
**Status:** Approved (design phase)
**Builds on:** the unified workbench (fusion graph + per-kernel drill + stage IR). This lights up the workbench **Tensor Diff** panel with L0 final-output diff (actual vs expected), sim backend.

## 1. Goal

When a network_runner work dir has been **executed** (sim) — i.e. it contains actual outputs and reference/expected `.npy` — `ingest` produces `summaries/tensor_diff.json` so the workbench Tensor Diff panel shows per-output PASS/FAIL + max/mean abs error. Out of scope: per-kernel first-bad locate (Phase 2b), NPU backend.

## 2. What already exists (verified)

- `examples/two-elewise-e2e/run.sh` runs `network_runner.py --backend sim` → writes `<work>/outputs/out0.npy`, `out1.npy` and logs `network.output[i]: max_diff=… PASS/FAIL` vs `<work>/expected0.npy`/`expected1.npy`. Verified working in this env (ASCEND_HOME_PATH=/home/gser/Ascend/cann).
- `python/tools/ascend_diff.py` has `diff_pair(actual_path, expected_path, atol, rtol) -> {max_diff, mean_diff, pass, ...}` (reusable; computes abs error + PASS/FAIL).
- Workbench `tensor_diff.json` schema (from `open_view._load_tensor_diff` + `_tensor_diff_rows` + `_overlay_summary`):
  ```json
  { "status": "PASS|FAIL", "comparison_count": N, "failed_count": M,
    "comparisons": [ { "status": "PASS|FAIL", "id": "network.output[0]",
                       "max_abs_error": <float>, "max_rel_error": <float>,
                       "mean_abs_error": <float>, "atol": <float>, "rtol": <float>,
                       "kernel_id": null, "task_id": null } ] } }
  ```

## 3. Design — `ingest` detects an executed run and writes tensor_diff.json

`ingest` already consumes a network_runner work dir. Add: if the run was executed, build the diff.

Detection + inputs (in the work dir):
- **Actual outputs**: `<workdir>/outputs/out*.npy` (sorted by the trailing index). If absent → run not executed → skip (no tensor_diff; panel stays empty, as today).
- **Expected**: `<workdir>/expected*.npy` (sorted to match output index). Pair positionally: `out<i>.npy` ↔ `expected<i>.npy`. (Two-elewise: `out0↔expected0`, `out1↔expected1`.) If counts mismatch or no expected → skip with a `log`.
- **atol/rtol**: not persisted by network_runner; default `atol=1e-2, rtol=1e-2` (matches the examples). (Optional later: read from a run-config if present.)

Build via `ascend_diff.diff_pair` per pair → one `comparison`:
`diff_pair` returns `{ok, kind, max_diff, mean_diff, atol, rtol, shape, first_mismatch, ...}` (PASS key is `ok`; no rel-error field):
```python
r = ascend_diff.diff_pair(actual_i, expected_i, atol=ATOL, rtol=RTOL)
comparisons.append({
    "status": "PASS" if r.get("ok") else "FAIL",
    "id": f"network.output[{i}]",
    "kernel_id": None, "task_id": None,
    "max_abs_error": r.get("max_diff"),
    "max_rel_error": None,                 # diff_pair does not compute rel error
    "mean_abs_error": r.get("mean_diff"),
    "atol": ATOL, "rtol": RTOL,
})
failed = sum(1 for c in comparisons if c["status"] != "PASS")
```
Then write `<run>/summaries/tensor_diff.json`:
```python
{ "schema_version": 1, "tool": "ascend-debug",
  "status": "PASS" if failed==0 else "FAIL",
  "comparison_count": len(comparisons),
  "failed_count": failed,
  "comparisons": comparisons }
```
`open_view.open_run` (already called at the end of ingest) → `_load_tensor_diff` reads it → `overlay_details.tensor_diff` → the workbench Tensor Diff panel renders.

Import: `ingest` adds `from python/tools` access to `ascend_diff` — since `ascend_diff.py` lives at `python/tools/ascend_diff.py` (repo `python/` on sys.path via PYTHONPATH), import as `from tools import ascend_diff` OR add its dir to `sys.path` and `import ascend_diff`. Confirm the import path during implementation (the tool already adds `<repo>/python` to sys.path). If importing is awkward, replicate the tiny `diff_pair` numpy logic inline (max abs / mean abs / allclose) — it's ~10 lines — to avoid a fragile cross-package import. Prefer reuse if clean.

## 4. Files

- Modify: `tools/ascend-debug/ascend_debug/ingest.py` — detect executed run, build + write `summaries/tensor_diff.json`.
- New: `tests/tools/ascend-debug/test_tensor_diff.py` — unit test the diff-builder helper with synthetic npy.
- No `open_view`/`debug_graph` changes (the Tensor Diff panel already consumes `tensor_diff.json`).

## 5. Testing

- Unit: write 2 pairs of tiny `.npy` (one matching, one diverging beyond atol) into a temp workdir layout (`outputs/out0.npy`+`expected0.npy`, `out1.npy`+`expected1.npy`); call the ingest diff-builder; assert `tensor_diff.json` has `comparison_count==2`, `failed_count==1`, correct per-output status + populated `max_abs_error`.
- Integration: run `examples/two-elewise-e2e/run.sh` (sim) to produce an executed `build_e2e`; `ingest build_e2e --out /tmp/te-wb`; assert `summaries/tensor_diff.json` exists with 2 PASS comparisons (max_abs_error ≈ 4.9e-4 / 9.8e-4); `open` workbench `overlay_details.tensor_diff.status == "PASS"`, `comparison_count==2`. Visual: Tensor Diff panel shows the 2 outputs PASS.
- Regression: existing pytest + lit pass; a non-executed ingest (outline-only, no outputs/) still produces no tensor_diff (panel empty) and doesn't error.

## 6. Risks

- **Output↔expected pairing** is positional by trailing index; if a network's output ordering differs, the pairing could be wrong. For the example flow (network_runner positional `--expected`) this matches. Note the assumption.
- **atol/rtol default** (1e-2) may not match a specific run's tolerance; acceptable for the panel (it shows the error magnitude; PASS/FAIL uses the default). A later enhancement can persist the run's tolerance.
- **ascend_diff import path** — resolve cleanly or inline the ~10-line numpy diff to avoid cross-package fragility.
- Phase 2b (per-kernel locate / first-bad) is separate and needs two per-kernel checkpoint sets; not here.
