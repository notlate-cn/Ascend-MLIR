# Runtime Scheduler Admission Cleanup Design

## Goal

Tighten the current `GlobalScheduler` admission implementation without changing
the external scheduler baseline.

This cleanup is specifically about:

- reducing repeated quota/priority/fairness scans
- making the admission decision path easier to read
- preserving the current shared observability contract

It is not a new scheduler-policy stage.

## Current State

The current scheduler baseline is already implemented and documented:

- stream-level resource accounting
- cross-session round-robin fairness
- static session priority
- session admission quota
- internal default policy surface

The main remaining issue is local implementation complexity inside
`GlobalScheduler.cpp`:

- session readiness and quota eligibility are scanned in multiple places
- quota-blocked ready-task marking is spread across multiple admission paths
- backend default capability construction is still inline in `submit(...)`

The code is correct enough to ship, but it has reached the point where another
policy layer would become harder to add cleanly.

## Non-Goals

This cleanup does not:

- change fairness semantics
- change quota semantics
- change priority semantics
- add CLI/env policy configuration
- introduce backend-specific policy branches
- change xvm verification scope

## Chosen Approach

Keep behavior stable and clean up the internal admission path around a few small
helpers.

The cleanup will:

1. add one more focused characterization test for default-policy vs explicit
   per-session scheduling
2. extract repeated admission predicates into small private helpers
3. extract backend default capability construction out of the submit overload
4. keep existing observability keys and meanings unchanged

## Expected Outcome

After the cleanup:

- the current scheduler baseline remains the same
- default policy and explicit per-session scheduling are both more obviously
  encoded in code
- the next scheduler stage can extend `GlobalScheduler` from a simpler internal
  structure instead of layering more logic on top of the current repeated scans
