# Runtime NPU Pre-Device Hardening Design

## Goal

Before NPU real-device hardware is available, tighten the runtime-side NPU
contract so that future bring-up failures are more likely to be true
driver/device issues rather than silent runtime contract drift.

## Scope

This design intentionally avoids new scheduler architecture or optimistic NPU
concurrency semantics. It only covers the pre-device hardening work that can be
validated on xvm today.

Included:

- `NpuBackend` request-contract validation
- focused NPU runtime tests for contract and driver-path behavior
- runtime architecture/audit docs for the current NPU baseline
- a real-device validation runbook that can be used once hardware is available

Excluded:

- new scheduler policy layers
- backend-specific scheduling branches
- real-device performance tuning
- claims of NPU completion without hardware

## Current Baseline

`NpuBackend` already supports two execution modes:

- driver-backed mode: delegate to `ExecutionBackendDriver`
- bare runtime mode: construct `RunArgs`, initialize a real-device execution
  runner, launch the binary or mix shared object, optionally compare against
  expected outputs, then write produced outputs

The main remaining risk is not architecture. It is contract ambiguity:

- request metadata can still be under-validated
- driver-backed paths do not have enough focused coverage
- the current README explains the scheduler baseline, but not the NPU bring-up
  checklist in enough operational detail

## Design

### 1. Tighten the runtime-owned NPU contract

`NpuBackend` should reject malformed runtime requests as early as possible,
before future real-device testing obscures failures behind lower-level errors.

The runtime-owned contract should explicitly cover:

- output binding count vs expected output count
- output binding metadata vs expected output metadata
- invalid external-file binding shapes/dtypes on the NPU path
- stable stage attribution for binding/runner/launch failures

This keeps `NpuBackend` conservative: runtime-owned validation happens in the
runtime layer, while driver/device behavior remains delegated.

### 2. Expand NPU-focused tests where real-device hardware is not required

The xvm baseline can already prove a meaningful subset of the NPU contract.
Focused tests should cover:

- malformed binding metadata
- driver delegation semantics
- driver error propagation
- bare NPU runner initialization stage attribution
- capability / scheduler contract invariants

The purpose is not to fake real-device completion. It is to ensure that once
hardware exists, failures are narrowed to the true unresolved layer.

### 3. Document the pre-device boundary explicitly

The runtime architecture docs should say exactly what is true today:

- scheduler integration is real
- xvm verifies wiring, contract, and mock/driver paths
- real-device validation is still missing

Separate audit/runbook documents should capture:

- current validated NPU behavior
- known pre-device gaps
- the exact bring-up sequence to run on real hardware

## Validation Plan

Fresh verification must continue to use the authoritative xvm baseline:

- `bash test/tools/runtime/run_runtime.sh`
- `bash test/tools/runtime/run_simbackend_examples.sh`
- `bash test/tools/examples/example_pipelines.sh`

## Success Criteria

This work is complete when:

- NPU-focused tests fail first, then pass with minimal implementation changes
- runtime docs clearly separate “code path wired” from “real-device validated”
- a bring-up runbook exists for the first real-device session
- xvm runtime verification remains green
