"""Challenge the four unit tests in isolated child interpreters, with a copy of the planted module."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
PKG = "research.activation_transfer.moment.dictionary_transfer.semantics."
CASES = [
    ("core", "trend = 7 * (q @ t)", "trend = (q @ t)", "patch_mean"),
    ("core", "r[\"id\"] not in used", "True", "training_signed"),
    ("core", "return np.maximum(0, fit[\"intercept\"] + fit[\"slope\"] * stat)",
             "return fit[\"intercept\"] + fit[\"slope\"] * stat", "training_signed"),
    ("evaluate", "rankdata(score[ix], axis=1)", "rankdata(score[ix], axis=1, method='min')", "tied_auc"),
    ("evaluate", "np.stack(groups)[draw].reshape(2000, -1)",
                  "np.random.default_rng(seed).integers(len(clusters), size=(2000, len(clusters)))", "tied_auc"),
    ("physical", "v-=v.mean(1,keepdims=True)", "v-=0", "physical_scores"),
]


def child(tag, path, test):
    name = PKG + tag
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    import pytest
    raise SystemExit(pytest.main(["-q", str(HERE / "test_semantics.py"), "-k", test]))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(*sys.argv[1:])
    for name, before, after, test in CASES:
        source = (HERE / (name + ".py")).read_text()
        assert source.count(before) == 1, before
        with TemporaryDirectory() as folder:
            path = Path(folder) / "research/activation_transfer/moment/dictionary_transfer/semantics" / (name + ".py")
            path.parent.mkdir(parents=True)
            path.write_text(source.replace(before, after))
            run = subprocess.run([sys.executable, "-m", PKG + "mutations", name, str(path), test],
                                 capture_output=True, text=True)
            if run.returncode != 1 or "1 failed" not in run.stdout or "AssertionError" not in run.stdout:
                raise RuntimeError(run.stdout + run.stderr)
        print(test, "<-", after, "1 assertion failed", flush=True)
