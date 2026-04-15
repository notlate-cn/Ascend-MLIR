# Runtime Mix Contract Boundaries

## Scope

- Object: runtime-native mix compile path.
- Semantics: direct-source is the default runtime artifact builder; legacy-preprocess is fallback.
- Result: callers consume compiled artifacts, not mix internals.

## Inputs

- Kernel source path and content.
- Kernel kind and entry name.
- SOC and Ascend/CANN toolchain path.
- Compile flags and include paths.
- Shape and tiling params.
- Runtime ABI and metadata schema.

## Outputs

- Device object/binary.
- `tiling.bin`.
- `blockDim`.
- Generated host stub.
- Runtime ABI metadata.
- Compile timing metadata.

## Current Boundary

- `ArtifactCompiler` dispatches `mix` to `MixDirectBackend`.
- `MixDirectBackend` owns direct-source and fallback-mode selection.
- `MixDirectCompilePipeline` owns staged compilation.
- `RuntimeFrontendCore`, `runtime-session`, C API, and execution backends should consume compiled artifact results only.

## Guardrails

- Do not leak mix preprocessing details into CLI, C API, or execution runners.
- Do not compare PyPTO frontend-only timing with full runtime artifact compile timing.
- Keep legacy-preprocess callable only as fallback while direct-source remains default.
