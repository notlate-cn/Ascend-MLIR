# Generic Mix Toolchain Design

## Goal

Replace the current sample-specific RuntimeMix flow with a generic, MLIR-driven mix toolchain that:

- does not require hardcoded sample kernel names
- does not require hardcoded tensor file names, shapes, or dtypes
- does not require example-specific ABI reconstruction in `MixDirectBackend`
- derives runtime launch and validation inputs from compiler artifacts

The target user workflow is:

1. Start from `step7_cann.mlir`
2. Generate an original mix kernel source from the lowering pipeline
3. Compile and pack the mix artifact
4. Materialize runtime metadata and IO ABI automatically
5. Convert `.npy` test data into runtime inputs automatically
6. Run simulator execution and compare outputs using the generated manifest

No manual `--name`, sample alias, or sidecar IO description file should be required.

## Non-Goals

- Generalizing non-mix Runtime support in this change
- Introducing a separate user-authored JSON ABI file
- Solving all possible legacy sample layouts at once
- Rewriting MLIR lowering itself

This design focuses on the RuntimeMix path and examples that already lower to valid mix kernels.

## Current Problems

The current RuntimeMix path works for `matmul-add-leakyrelu`, but only after several sample-specific accommodations:

- `MixDirectBackend.cpp` reconstructs ABI through `buildCurrentSampleAbi(...)`
- sample names like `fc_leakyrelu`, `fc_relu_split`, and `baremix_custom` drive special behavior
- example scripts write multiple alias filenames to satisfy runtime expectations
- runtime-generated runner code assumes known tensor ordering and output file naming
- some kernel source handling depends on ad hoc source materialization rules

This has two structural consequences:

1. The runtime layer is not truly reusable across new mix kernels.
2. The example layer must preserve duplicated compatibility filenames and kernel aliases.

## Design Principles

### 1. `step7_cann.mlir` is the ABI source of truth

Static kernel ABI should come from MLIR, not from sample-name branches in runtime.

The extracted ABI must cover:

- logical kernel name
- argument list and ordering
- tensor direction: input vs output
- tensor shape
- tensor dtype
- presence and position of workspace and tiling arguments

### 2. Toolkit-generated preprocess outputs are the source of executable naming

Executable runtime symbols should come from toolkit-generated artifacts, not from user-entered names or handwritten alias maps.

These artifacts already define:

- launcher header name
- launcher symbol
- generated device source path
- AIC/AIV compile definitions
- final mix entry names such as `*_0_mix_aic` and `*_0_mix_aiv`

### 3. Launch metadata remains post-compile information

Some runtime metadata only becomes reliable after compilation and tiling generation. In particular:

- block dimension
- final launcher symbol
- final packed kernel identity

This information should be emitted by the compiled artifact and merged into the final manifest.

### 4. Example scripts become orchestration only

Examples should no longer reconstruct runtime ABI manually. They should only:

- run the MLIR pipeline
- invoke the generic mix compiler
- generate or load `.npy`
- invoke generic conversion from `.npy` to runtime `.bin`
- run the generic validator

## Proposed Architecture

### A. `MixAbi` data model

Introduce a single runtime ABI struct that represents everything needed for:

- input materialization
- output expectation
- runner generation
- validator setup

Suggested fields:

- `logicalKernelName`
- `runtimeKernelName`
- `arguments`
- `inputs`
- `outputs`
- `workspaceArgIndex`
- `tilingArgIndex`
- `workspaceBytes`
- `blockDim`
- `tilingMode`
- `tilingSource`
- `launcherSymbol`
- `aicEntry`
- `aivEntry`

Tensor descriptors should carry:

- name
- dtype
- shape
- canonical runtime filename

### B. MLIR ABI extraction pass/tool

Add a utility that reads `step7_cann.mlir` and extracts a `MixAbi` skeleton before runtime compilation.

This extractor should infer:

- kernel logical name from the CANN-signature function
- tensor arguments from the function signature
- directionality from attributes or signature conventions
- tensor filenames using a canonical rule, not sample aliases

Recommended filename rule:

- inputs: `<kernel>.<tensor>.input.bin`
- outputs: `<kernel>.<tensor>.output.bin`
- golden outputs: `<kernel>.<tensor>.golden.bin`

This removes the need for duplicated filenames like both `matmul_add_leakyrelu_input_a.bin` and `fc_leakyrelu_input_a.bin`.

### C. Original kernel source materialization

RuntimeMix must compile an original `__global__` kernel source, not a wrapperless inline-only variant.

The generic rule should be:

- runtime input source must preserve the original kernel entry
- toolkit preprocess must generate the `auto_gen_*` wrapper and launcher
- runtime should not maintain sample-name-specific source canonicalization rules

If a source transformation is required, it must be structural and generic, for example:

- normalize include paths
- normalize output source location
- preserve kernel entry unchanged

It must not be based on kernel names like `fc_relu_split` or `fc_leakyrelu`.

### D. Artifact introspection

After preprocess and extract-host-stub, runtime should collect:

- generated source path
- launcher header path
- launcher symbol
- AIC compile definitions
- AIV compile definitions

After runner tiling emission, runtime should collect:

- `block_dim`

These values should be merged into the `MixAbi` manifest and become the only source consumed by validation.

### E. Generic manifest

The final manifest should be promoted from debug output to the official contract between:

- compiler
- runner
- validator
- example data preparation

The manifest should contain:

- kernel identity
- tensor ABI
- runtime file naming
- workspace bytes
- block dimension
- tiling source
- launcher identity
- compile/generated artifact references for debugging

The validator and data-conversion code should consume this manifest directly.

### F. Generic example data bridge

Provide a helper that:

1. reads the manifest
2. reads `.npy` files from a directory
3. matches them to ABI tensor names
4. writes canonical `.bin` files into the runtime input/output directories

For examples generated from MLIR, `gen_data.py` should only produce:

- `input_*.npy`
- `output.npy` or named output `.npy`

Then the helper maps them into canonical runtime names.

This avoids per-example file duplication logic.

## Concrete Refactoring Targets

### Remove from `MixDirectBackend.cpp`

- `buildCurrentSampleAbi(...)`
- `isManualGeneratedSampleKernel(...)`
- `isSplitReluSampleKernel(...)`
- `canonicalizeSampleRuntimeKernelName(...)`
- `resolveSplitReluHostSourcePath(...)`
- `materializeSplitReluCanonicalDeviceSource(...)`

These functions should be replaced with:

- MLIR ABI extraction
- artifact introspection
- generic source/materialization helpers

### Keep and generalize

- launch info emission
- manifest writing
- generated compile-definition propagation
- runner generation

### Update `mix-validator`

`mix-validator` should continue to consume the manifest, but it should stop assuming:

- sample-specific filenames
- sample-specific tensor naming

It should rely entirely on:

- `abi_input*`
- `abi_output*`
- `abi_workspace_bytes`
- `abi_block_dim`
- `abi_tiling_*`

### Update examples

`examples/matmul-add-leakyrelu/run.sh` should move to:

- generate `step7_cann.mlir`
- generate original kernel source
- call generic mix compile
- call generic `npy -> bin` bridge
- call validator

No hardcoded alternate filenames should remain in the example.

## Data Flow

### Compile-time flow

1. MLIR pipeline emits `step7_cann.mlir`
2. ABI extractor builds initial `MixAbi`
3. Source materializer produces original runtime kernel source
4. RuntimeMix preprocesses source
5. Toolkit emits generated source, launcher header, host stub, AIC/AIV config
6. RuntimeMix compiles AIC and AIV with generated compile definitions
7. RuntimeMix links and packs artifact
8. Runner emits tiling blob and launch info
9. RuntimeMix writes final manifest from `MixAbi + artifact metadata + launch metadata`

### Validation flow

1. Data bridge reads manifest
2. Data bridge converts `.npy` into canonical `.bin` input files
3. Validator reads manifest
4. Validator loads packed artifact, tiling, and canonical input files
5. Validator writes canonical actual output files
6. Validator compares actual outputs with canonical golden outputs

## Filename and Naming Rules

To remove ambiguity, the generic toolchain should define canonical names.

### Kernel naming

- `logicalKernelName`: derived from MLIR
- `runtimeKernelName`: derived from toolkit launcher identity if different

### Tensor file naming

Per tensor:

- input: `<kernel>.<tensor>.input.bin`
- actual output: `<kernel>.<tensor>.output.bin`
- golden output: `<kernel>.<tensor>.golden.bin`

This should be emitted into the manifest so callers do not reconstruct names.

## Error Handling

The generic path should fail early when:

- MLIR ABI extraction cannot determine tensor direction or shape
- runtime source does not contain a valid original kernel entry
- preprocess does not produce launcher metadata
- AIC/AIV compile definitions are empty for a mix kernel
- launch info is missing
- `.npy` data cannot be matched to ABI tensor names

Errors should mention:

- source file or MLIR path
- expected field
- actual missing or invalid artifact

## Testing Strategy

### Unit-level

- ABI extraction from `step7_cann.mlir`
- tensor filename canonicalization
- manifest round-trip parse
- `.npy` to canonical `.bin` conversion

### Integration-level

- `examples/matmul-add-leakyrelu/run.sh`
- existing `fc_relu_split` mix example after migration

### Regression expectations

- no sample-name hardcoding remains in runtime
- packed artifact produces correct `*_0_mix_aic/aiv` entries
- validator result matches golden outputs exactly for known examples

## Migration Plan

### Phase 1

Introduce MLIR ABI extraction and canonical manifest generation while preserving current runner and validator behavior.

### Phase 2

Migrate `matmul-add-leakyrelu/run.sh` to canonical filenames and generic data bridge.

### Phase 3

Delete sample-specific runtime branches after at least two mix examples run through the generic path.

## Open Decisions

1. Exact MLIR location for tensor direction metadata

This design assumes `step7_cann.mlir` carries enough information to distinguish input and output tensors. If not, the lowering pipeline must add explicit attributes before runtime refactoring.

2. Mapping `.npy` names to tensor names

Preferred rule:

- if `.npy` filename matches tensor name, use it directly
- otherwise support a small deterministic mapping for `input_*.npy` and `output.npy`

This is still generic because it is based on tensor position and direction, not kernel name.

3. Multi-output kernels

The manifest and data bridge should support multiple outputs from the start, even if current examples only use one.

## Recommendation

Implement the generic path in the runtime first, then migrate examples to consume it. The runtime contract must stabilize before simplifying example scripts.

The key success criterion is not just that `matmul-add-leakyrelu` still passes, but that a second mix example can be migrated without adding any new kernel-name branch in runtime.
