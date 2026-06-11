import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "ascend-debug"))

from ascend_debug import collect


def test_six_stages_in_order():
    names = [s[0] for s in collect.PASS_STEPS]
    assert names == [
        "normalize", "schedule", "bufferize",
        "realize", "parallelize", "finalize",
    ]


def test_stage_io_chains():
    steps = collect.PASS_STEPS
    assert steps[0][1] == "source"
    for prev, cur in zip(steps, steps[1:]):
        assert cur[1] == prev[2], (prev, cur)


def test_no_outline_or_devnyh_flags():
    for _, _, _, flags in collect.PASS_STEPS:
        joined = " ".join(flags)
        # outlining belongs to --auto-fuse, not --auto-fuse-codegen
        assert "--auto-fuse-group-analysis" not in joined
        assert "--auto-fuse-group-outline" not in joined
        # dev-nyh pass/option names must not appear
        assert "--ascend-normalize" not in joined
        assert "debug-dump-dir" not in joined
        assert "dump-report" not in joined


def test_key_develop_flags_present():
    flags = dict((s[0], s[3]) for s in collect.PASS_STEPS)
    assert "--auto-fuse-tile-fuse" in flags["schedule"]
    assert any("one-shot-bufferize" in f for f in flags["bufferize"])
    assert "--linalg-to-ascendc" in flags["realize"]
    assert "--canonicalize-cann-signature" in flags["finalize"]
