#!/usr/bin/env bash
# Mirror of examples/bert-e2e/env_sibling.sh — point tool paths at the in-tree
# build/bin so this example can be driven without a separate worktree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "${WT_ROOT}/examples/env.sh"

SIBLING_BIN="${WT_ROOT}/build/bin"
export AFIR_OPT="${SIBLING_BIN}/afir-opt"
export AFIR_TRANSLATE="${SIBLING_BIN}/afir-translate"
export ACLNN_BACKEND="${SIBLING_BIN}/aclnn-backend"
export RUNTIME_SESSION="${SIBLING_BIN}/runtime-session"
export AUTOTUNER="${SIBLING_BIN}/autotuner"
export PATH="${SIBLING_BIN}:${PATH}"
echo "gpt2-e2e tools -> ${SIBLING_BIN}"
