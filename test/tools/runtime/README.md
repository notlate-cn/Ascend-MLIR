# Runtime Test And Diagnostic Tools

## Default Verification

- Object: `run_runtime.sh`.
- Semantics: focused runtime verification on xvm.
- Result: keeps runtime core, C API, and repeated mix simulation baseline green.

## Mix Timing Diagnostics

| Tool | Semantics | Default Verification |
| --- | --- | --- |
| `run_mix_compile_timing_compare.sh` | Ascend-MLIR direct-source compile-only timing baseline | no |
| `run_mix_pypto_timing_compare.sh` | Ascend-MLIR timing plus PyPTO frontend/public timing comparison | no |
| `pypto_mix_public_breakdown.py` | PyPTO public call breakdown into Python wrapper, compile, and `_run_with_cpu` | no |

## xvm Commands

```bash
cd /home/niu/code/Codex-Ascend-MLIR
bash test/tools/runtime/run_runtime.sh

RUNS=3 bash test/tools/runtime/run_mix_compile_timing_compare.sh
RUNS=3 bash test/tools/runtime/run_mix_pypto_timing_compare.sh

cd /home/niu/code/pypto
PYTHONPATH=/home/niu/code/pypto/python \
  python3 /home/niu/code/Codex-Ascend-MLIR/test/tools/runtime/pypto_mix_public_breakdown.py --runs 3
```

## Interpretation Rules

- `run_mix_compile_timing_compare.sh` reports Ascend-MLIR direct-source compile-only timing.
- `pypto_frontend_compile_private` reports PyPTO private frontend/JIT compile boundary.
- `pypto_compile_and_sim_public` reports PyPTO public compile+SIM/cost-model/report timing.
- `pypto_mix_public_breakdown.py` explains why PyPTO public timing is dominated by `_run_with_cpu`.
