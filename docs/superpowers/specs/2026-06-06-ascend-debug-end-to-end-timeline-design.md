# Ascend Debug End-to-End Timeline Design

## Status

Approved design direction on 2026-06-06. This document covers the UI and data-model design only; implementation requires a follow-up plan.

## Problem

`ascend-debug open` currently renders Stage Timeline and Kernel / Runtime Artifacts as adjacent but separate surfaces. Users can inspect stage dumps, reports, `kernel.cpp`, `host_tiling.cpp`, `artifact_manifest.json`, and related summaries, but the UI does not explain which compiler step produced which runtime artifact or which stable debug contract should be used as the semantic source.

This weakens the "one-stop" debugging workflow because a user must manually jump between the Stage Timeline, graph view, artifact manifest dashboard, raw artifacts, and reports.

## Goals

- Present a single end-to-end timeline from input IR to final runtime artifacts.
- Keep stage dumps as source browsing and fallback evidence, not the primary semantic interface.
- Prefer versioned Debug Contract JSON when present.
- Preserve compatibility with current `manifest.json`, `artifact_manifest.json`, and `summaries/debug_graph.json`.
- Make unknown or future artifact fields visible as raw data rather than blocking the tool.
- Avoid duplicating pass internals across compiler code and Python UI logic.

## Non-Goals

- Do not redesign the graph canvas in this change.
- Do not require all Debug Contract JSON files to exist before the UI improves.
- Do not parse additional stage MLIR text to infer runtime semantics.
- Do not add new field-level parsing for every Ascend pass attribute.

## Recommended Approach

Use approach B: upgrade the current Stage Timeline into an end-to-end main lane.

The visible lane is:

```text
Source -> Normalize -> Kernelize -> Schedule -> Realize -> Translate -> Artifacts -> Runtime
```

Each lane item displays three categories:

- Stage Evidence: MLIR dump, graph view, command, stdout/stderr, pass report.
- Semantic Contract: stable contract JSON when present, with a visible source label such as `contract` or `legacy_adapter`.
- Produced Artifacts: generated runtime/kernel files and dashboards linked to the producing lane.

## Timeline Model

Introduce a Python-side `timeline_model` as the only UI input for the timeline renderer.

Inputs:

- `manifest.json`
- `artifact_manifest.json`
- optional `stage_manifest.json`
- optional `kernel_dag.json`
- optional `schedule_decisions.json`
- optional `memory_plan.json`
- optional `run_manifest.json`
- existing summaries such as `summaries/debug_graph.json`, tensor diff, locate, and memory summaries

Output shape:

```json
{
  "schema_version": 1,
  "source": "contract|legacy_adapter|mixed",
  "lanes": [
    {
      "id": "translate",
      "title": "Translate",
      "stage_orders": [60, 70, 80, 90],
      "evidence": [],
      "contracts": [],
      "artifacts": [],
      "diagnostics": []
    }
  ]
}
```

The UI should render only this model. Compatibility logic stays in the model builder.

## Artifact Attachment Rules

First version attachment is deterministic and conservative:

- `kernel_dag` and graph summaries attach to `Kernelize`.
- `schedule_decisions` and `tiling_space.json` attach to `Schedule`.
- `memory_plan` and memory summaries attach to `Realize`.
- `kernel.cpp` and `host_tiling.cpp` attach to `Translate`.
- `artifact_manifest.json` attaches to `Artifacts`.
- `run_manifest.json`, run status, tensor diff, and locate summaries attach to `Runtime`.

When a future manifest entry contains explicit fields such as `producer_stage`, `producer_step`, `contract_kind`, or `source_contract`, those fields take precedence over kind-based mapping.

## UI Behavior

The homepage should keep Kernel / Runtime Artifacts visible, but the primary navigation becomes the timeline:

- Each lane item shows compact counts for dumps, reports, contracts, and artifacts.
- Each attached artifact exposes Raw and View links.
- Contract-backed entries show a `contract` badge.
- Compatibility-derived entries show a `legacy_adapter` badge.
- Missing optional contracts show "not emitted" rather than an error.
- Unknown artifact fields are shown in a raw detail drawer.

The existing artifact manifest dashboard remains the detailed artifact view. The timeline links into it instead of duplicating the full manifest table.

## Debug Contract Migration

The timeline model is the bridge between today's artifacts and the long-term Debug Contract:

1. Build the model from current manifests and summaries.
2. Teach the model builder to prefer explicit contract files when present.
3. Add compiler-side manifest fields for artifact provenance.
4. Remove kind-based compatibility rules only after contract coverage is complete and tested.

Expected future compiler-side fields:

```json
{
  "path": "kernel.cpp",
  "kind": "kernel-cpp",
  "producer_stage": "translate",
  "producer_step": "ascend-canonicalize-cann-signature",
  "contract_kind": "cann_kernel_source",
  "source_contract": "artifact_manifest.json"
}
```

## Testing

Add focused tests around the model builder and rendered homepage:

- Current deep collect output links `kernel.cpp` and `host_tiling.cpp` to `Translate`.
- `artifact_manifest.json` is reachable from the `Artifacts` lane.
- Missing optional contract files do not fail rendering.
- Explicit `producer_stage` overrides kind-based fallback.
- Unknown artifact fields are preserved in raw detail output.
- Rendered HTML contains `legacy_adapter` when compatibility data is used.

Browser verification should check:

- The homepage opens.
- The end-to-end lane is visible.
- Artifact links navigate to raw and view pages.
- The debug graph page still opens from timeline stage entries.

## Risks

- If the first version over-indexes on kind-based mapping, it can become another hidden contract. Keep the mapping small and clearly marked as compatibility-only.
- If the UI duplicates artifact manifest details, the homepage will become noisy. Keep the homepage as a navigation and summary surface.
- If contract and compatibility data conflict, contract data should win and the UI should expose the conflict as a diagnostic.
