# Runtime Mix Contract Boundaries

## Scope

- Object: runtime-native mix compile path.
- Semantics: direct-source is the only runtime artifact builder.
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
- `MixDirectBackend` owns direct-source artifact construction.
- `MixDirectCompilePipeline` owns staged compilation.
- `RuntimeFrontendCore`, `runtime-session`, C API, and execution backends should consume compiled artifact results only.

## Guardrails

- Do not leak mix preprocessing details into CLI, C API, or execution runners.
- Do not compare PyPTO frontend-only timing with full runtime artifact compile timing.
- Do not reintroduce legacy-preprocess fallback.
