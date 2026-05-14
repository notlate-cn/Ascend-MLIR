#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if sed -n '/struct ScheduleDecision {/,/};/p' \
    include/Conversion/Ascend/Schedule/ScheduleTypes.h |
    grep -E 'SmallVector<ScheduleGuard> +(candidateGuards|decisionGuards)' \
      >/dev/null; then
  echo "ScheduleDecision must not duplicate ScheduleInstance guard vectors" >&2
  exit 1
fi

if grep -R -nE 'decision[.]candidateGuards|decision[.]decisionGuards|selectedDecision[.]candidateGuards|selectedDecision[.]decisionGuards' \
    include/Conversion/Ascend lib/Conversion/Ascend test/Conversion \
      >/dev/null; then
  echo "ScheduleDecision guard users must read through decision.instance" >&2
  exit 1
fi
