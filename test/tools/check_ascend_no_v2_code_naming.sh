#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if rg -n \
  'AscendV2|ascend\.v2|::v2\b|\bv2::|Ascend V2|ascend-v2-pipeline' \
  include/Conversion lib/Conversion tools/afir-opt test/Conversion test/unittests/Conversion \
  --glob '!build/**'; then
  echo "found V2/v2 code naming in Ascend pipeline sources" >&2
  exit 1
fi
