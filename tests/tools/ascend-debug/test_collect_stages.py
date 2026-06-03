import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "ascend-debug"))

from ascend_debug import collect


def test_six_stages_in_order():
    names = [s[0] for s in collect.PASS_STEPS]
    assert names == [
        "normalize", "kernelize", "schedule",
        "realize", "parallelize", "finalize",
    ]


def test_stage_io_chains():
    steps = collect.PASS_STEPS
    assert steps[0][1] == "source"
    for prev, cur in zip(steps, steps[1:]):
        assert cur[1] == prev[2], (prev, cur)


def test_no_devnyh_pass_flags():
    for _, _, _, flags in collect.PASS_STEPS:
        joined = " ".join(flags)
        assert "--ascend-normalize" not in joined
        assert "debug-dump-dir" not in joined
        assert "dump-report" not in joined


def test_normalize_uses_develop_flags():
    flags = dict((s[0], s[3]) for s in collect.PASS_STEPS)
    assert "--auto-fuse-group-analysis" in flags["kernelize"]
    assert "--linalg-to-ascendc" in flags["realize"]
    assert "--canonicalize-cann-signature" in flags["finalize"]
