# Ascend Debug Contract v1

Ascend Debug Contract files are versioned JSON files produced by the compiler
or by a compatibility adapter. They are the primary semantic interface for
`ascend-debug`. Stage MLIR dumps and pass reports are source browsing and
legacy fallback inputs only.

Every contract file is a JSON object with:

```json
{
  "schema": "ascend.debug.<name>",
  "schema_version": 1,
  "producer": {
    "tool": "ascend-mlir-opt",
    "pass": "ascend-schedule"
  },
  "data": {}
}
```

Required v1 contracts:

- `stage_manifest.json`: ordered stage records, step catalog entries, report
  paths, graph paths, and contract paths.
- `schedule_decisions.json`: schedule decisions by kernel with guards,
  fallback markers, tile summaries, and raw extension payloads.
- `kernel_dag.json`: kernel DAG nodes and edges with stable display facts and
  optional raw runtime-artifact references.
- `memory_plan.json`: realize memory placement, movement, workspace slots, and
  failure reasons.

Unknown fields are preserved under `raw` or `extensions` and must not make the
debug UI fail.
