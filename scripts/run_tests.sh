#!/bin/bash
#===----------------------------------------------------------------------===//
# Test Runner Script for Ascend-MLIR
#===----------------------------------------------------------------------===//

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

echo "=== Running Ascend-MLIR Tests ==="

if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: Build directory not found. Please build the project first."
    exit 1
fi

cd "${BUILD_DIR}"

# Run lit tests
echo "Running lit tests..."

START_TIME=$(date +%s)

cmake --build . --target check-afir

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_SECONDS / 60))
REMAINING_SECONDS=$((TOTAL_SECONDS % 60))

echo "=== Tests completed in ${MINUTES}m${REMAINING_SECONDS}s ==="
