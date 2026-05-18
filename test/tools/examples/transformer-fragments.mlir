// REQUIRES: ascend_env
// RUN: bash %S/../../../examples/transformer-fragments/run-mainline.sh --fragment layernorm --log | FileCheck --check-prefix=LAYER %s
// RUN: timeout 30s bash %S/../../../examples/transformer-fragments/run-mainline.sh --fragment qkv --runtime-e2e --log | FileCheck --check-prefix=QKV %s

// LAYER: transformer_fragment.layernorm.full_codegen=pass
// LAYER: transformer_fragment.layernorm.phase5_translate=pass
// LAYER: transformer_fragment.layernorm.runtime_session=pass
// LAYER: transformer_fragment.layernorm.validation=pass

// QKV: transformer_fragment.qkv.full_codegen=pass
// QKV: transformer_fragment.qkv.phase5_translate=pass
// QKV: transformer_fragment.qkv.artifact_compile=pass
// QKV: transformer_fragment.qkv.runtime_session=pass
// QKV: transformer_fragment.qkv.validation=pass
