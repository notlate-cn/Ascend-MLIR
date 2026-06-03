import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "tools" / "ascend-debug"
sys.path.insert(0, str(PKG_DIR))


def _load_entry():
    spec = importlib.util.spec_from_file_location(
        "ascend_debug_cli", PKG_DIR / "ascend-debug.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_has_only_collect_and_open():
    entry = _load_entry()
    parser = entry.build_parser()
    sub = next(a for a in parser._actions if hasattr(a, "choices") and a.choices)
    assert set(sub.choices) == {"collect", "open"}


def test_collect_requires_input_and_out():
    entry = _load_entry()
    parser = entry.build_parser()
    args = parser.parse_args(["collect", "in.mlir", "--out", "run"])
    assert str(args.input) == "in.mlir"
    assert str(args.out) == "run"
