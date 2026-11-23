"""Challenge the two unit tests in isolated child interpreters, with a copy of the planted module."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
PKG = "research.activation_transfer.moment.dictionary_transfer.input_deltas.template."
CASES = [
    ("run","np.asarray(history,float).mean(axis=0)","np.asarray(mapped,float).mean(axis=0)","historical_template"),
    ("run","lengths[:,None]*(template/size)[None]","np.broadcast_to(template,mapped.shape).copy()","historical_template"),
    ("score","describe(skills[a]-skills[b])","describe(skills[b]-skills[a])","paired_positive"),
    ("score","vals-vals.mean(1,keepdims=True) if mode==\"shape\" else vals","vals","paired_positive"),
]


def child(tag, path, test):
    name = PKG + tag
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    import pytest
    raise SystemExit(pytest.main(["-q", str(HERE / "test_template.py"), "-k", test]))


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
