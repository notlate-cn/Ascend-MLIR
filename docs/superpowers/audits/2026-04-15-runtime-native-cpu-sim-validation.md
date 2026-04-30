# Runtime-Native CPU-Sim Validation

Date: 2026-04-15

## Scope

Object: runtime-native CPU simulation path.

Confirmed semantics: compile and execute vec/mix artifacts through the current
runtime center without NPU hardware.

Result: xvm CPU-sim validation is green.

## Verified Baselines

Object: `test/tools/runtime/run_runtime.sh`.

Confirmed semantics: focused runtime verification, C API runtime path,
runtime-session planning/negative paths, SimBackend smoke, repeated mix
simulation.

Result:

- `test_taskgraph_runtime`: `554 passed, 0 failed`
- `test_capi_runtime`: `15 passed, 0 failed`
- `test_runtime`: `109 passed, 0 failed`
- SimBackend smoke baseline: pass
- repeated mix simulation baseline: pass

Object: `test/tools/runtime/run_simbackend_examples.sh`.

Confirmed semantics: runtime-session CPU-sim execution across vec and mix
examples.

Result:

- `relu-broadcast-transpose`: pass
- `add-broadcast-concat`: pass
- `broadcast-add-reduce`: pass
- `gather-elementwise-fusion`: pass
- `split-relu-brc-add-mul`: pass
- `matmul-add-leakyrelu`: pass

Object: `test/tools/examples/example_pipelines.sh`.

Confirmed semantics: six example pipelines compile and execute end-to-end.

Result: `ALL EXAMPLE PIPELINES PASSED`.

## Runtime-Native State

Object: runtime compile entry.

Confirmed semantics: `runtime-session` remains the general runtime CLI;
`MixDirectBackend` enters the direct mix compile pipeline through
`executeMixDirectCompile`.

Result: public compile path no longer depends on `Legacy/Compiler`.

Object: direct mix compile implementation.

Confirmed semantics: direct mix compile is split by stage.

Result:

- `MixDirectCompilePipeline.cpp`: orchestration and artifact finalization
- `MixDirectCompileLayout.cpp`: artifact/workdir layout
- `MixDirectPreprocess.cpp`: probe, preprocess, generated config, compile contract
- `MixDirectBinaryBuild.cpp`: device/host binary build and packed shared object
- `MixDirectRuntimeAbi.cpp`: ABI extraction, concrete shape reconciliation, metadata
- `MixDirectTilingArtifacts.cpp`: tiling binary and launch info generation
- `MixDirectCompileSupport.cpp`: shared process/file helpers

Object: NPU real-device validation.

Confirmed semantics: not executed because no NPU device is available.

Result: remaining original-task gap is real-device validation only.
