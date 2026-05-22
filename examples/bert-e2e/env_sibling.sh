#!/usr/bin/env bash
# Source examples/env.sh for sim LD_LIBRARY_PATH, then override tool paths to the
# sibling encoder-robustness build (same dev-network commit 5f2c754e) since this
# worktree has no build/ of its own.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "${WT_ROOT}/examples/env.sh"

SIBLING_BIN="/home/gser/code/Ascend-MLIR/.claude/worktrees/encoder-robustness/build/bin"
export AFIR_OPT="${SIBLING_BIN}/afir-opt"
export AFIR_TRANSLATE="${SIBLING_BIN}/afir-translate"
export ACLNN_BACKEND="${SIBLING_BIN}/aclnn-backend"
export RUNTIME_SESSION="${SIBLING_BIN}/runtime-session"
export AUTOTUNER="${SIBLING_BIN}/autotuner"
export PATH="${SIBLING_BIN}:${PATH}"
echo "BERT-e2e tools -> ${SIBLING_BIN}"
