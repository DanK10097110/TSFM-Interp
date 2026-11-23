"""Challenge the three unit tests in isolated child interpreters, with a copy of the planted module."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
PKG = "research.activation_transfer.moment.forecast_probe."
CASES = [
    ("capture", "context[:, :-16]", "context[:, 16:]", "withheld_labels"),
    ("capture", "mask[:, :pad] = 0", "mask[:, -pad:] = 0", "withheld_labels"),
    ("capture", 'mode=kind', 'mode="constant"', "withheld_labels"),
    ("fit", "mean_x = x[train].mean(0)", "mean_x = x.mean(0)", "dual_matches"),
    ("fit", "scale * precision", "precision", "dual_matches"),
    ("fit", "value += (last_raw - mu)[:, None]", "value += last_raw[:, None]", "restores_native"),
]


def child(tag, path, test):
    name = PKG + tag
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    import pytest
    raise SystemExit(pytest.main(["-q", str(HERE / "test_forecast_probe.py"), "-k", test]))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(*sys.argv[1:])
    for name, before, after, test in CASES:
        source = (HERE / (name + ".py")).read_text()
        assert source.count(before) == 1, before
        with TemporaryDirectory() as folder:
            path = Path(folder) / (name + ".py")
            path.write_text(source.replace(before, after))
            run = subprocess.run([sys.executable, "-m", PKG + "mutations", name, str(path), test],
                                 capture_output=True, text=True)
            if run.returncode != 1 or "1 failed" not in run.stdout or "AssertionError" not in run.stdout:
                raise RuntimeError(run.stdout + run.stderr)
        print(test, "<-", after, "1 assertion failed", flush=True)
