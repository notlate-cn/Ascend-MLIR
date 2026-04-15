# PyPTO Mix Timing Semantics

## Scope

- Object: PyPTO mix timing comparison on xvm.
- Semantics: distinguish frontend compile, public compile+SIM, and public-call breakdown.
- Result: prevent comparing PyPTO private frontend timing directly against Ascend-MLIR runtime artifact compile timing.

## Measured Cases

| Object | Semantics | xvm Result |
| --- | --- | --- |
| `pypto_frontend_compile_private` | private frontend/JIT compile boundary | about `130-150 ms` |
| `pypto_compile_and_sim_public` | public call: compile + SIM/cost-model/report | about `2.2-2.3 s` |
| `pypto public breakdown compile` | compile portion inside public call | about `90-170 ms` |
| `pypto public breakdown _run_with_cpu` | SIM/cost-model/report portion inside public call | about `2.0-2.2 s` |

## Interpretation

- PyPTO public `2.2 s` is dominated by `_run_with_cpu`.
- PyPTO private frontend timing is not a full artifact compile timing.
- Ascend-MLIR direct-source compile timing remains a runtime artifact compile timing.

## Reproduction

Run on xvm:

```bash
cd /home/niu/code/Codex-Ascend-MLIR
bash test/tools/runtime/run_mix_pypto_timing_compare.sh
cd /home/niu/code/pypto
PYTHONPATH=/home/niu/code/pypto/python \
  python3 /home/niu/code/Codex-Ascend-MLIR/test/tools/runtime/pypto_mix_public_breakdown.py --runs 3
```

## Decision

- Use `pypto_frontend_compile_private` only as a frontend-lightweight reference.
- Use `pypto_compile_and_sim_public` only as user-visible SIM invocation timing.
- Do not compare either directly against Ascend-MLIR compile-only timing without naming the scope.
