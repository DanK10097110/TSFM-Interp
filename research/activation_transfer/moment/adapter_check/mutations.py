"""Verify the adapter's numerical regression assertions by running isolated, deliberately incorrect implementations."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / "tsfm_lens/tsfm_lens/models/moment_adapter.py"
SUITE = ROOT / "tsfm_lens/tests/test_moment_adapter.py"
PLANTS = (
    ("Raw units", "padding_missing_values", "values = torch.where(present, x, torch.nan)",
     "values = torch.zeros_like(x)"),
    ("Missing value replaced before native embedding", "padding_missing_values", "values = torch.where(present, x, torch.nan)",
     "values = torch.where(present, x, torch.zeros_like(x))"),
    ("Padding included in RevIN", "padding_missing_values", "F.pad(present.long(), (pad, 0))",
     "F.pad(present.long(), (pad, 0), value=1)"),
    ("Fractional future patch dropped", "history_geometry", "math.ceil(self._first_horizon(horizon) / self._patch)",
     "self._first_horizon(horizon) // self._patch"),
    ("Partial history patch retained", "history_geometry", "context_len // self._patch",
     "math.ceil(context_len / self._patch)"),
    ("Wrong crop edge", "long_rollout", "x = x[:, -self._width:]", "x = x[:, :self._width]"),
    ("Missing rollout feedback", "long_rollout", "values[:, :, count:], pred[:, None]",
     "values[:, :, count:], torch.zeros_like(pred[:, None])"),
    ("Feedback mean detached", "rollout_gradient", "d_mean = mean - mean.detach()",
     "d_mean = torch.zeros_like(mean)"),
    ("Feedback scale detached", "rollout_gradient", "d_scale = scale - scale.detach()",
     "d_scale = torch.zeros_like(scale)"),
    ("Unpinned default model", "load_requests", 'self.cfg.kwargs.get("revision", self.published_revisions.get(checkpoint))',
     'self.cfg.kwargs.get("revision")'),
    ("Random task head requested", "load_requests", 'model_kwargs={"task_name": "reconstruction"}',
     'model_kwargs={"task_name": "forecasting"}'),
)


def main():
    """Replace just the class under test, preserving the normal registry route."""
    with tempfile.TemporaryDirectory(prefix="moment_known_wrong_") as temp:
        directory=Path(temp)
        initial=SUITE.read_text()
        loader="""from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tsfm_lens.models import ADAPTERS
spec = spec_from_file_location("tsfm_lens.models.moment_planted", Path(__file__).with_name("mutant.py"))
mutated = module_from_spec(spec)
spec.loader.exec_module(mutated)
MomentAdapter = mutated.MomentAdapter
ADAPTERS["moment"] = MomentAdapter"""
        suite=initial.replace("from tsfm_lens.models.moment_adapter import MomentAdapter",loader)
        for i,(name,test,old,new) in enumerate(PLANTS):
            parent=directory/str(i);parent.mkdir()
            impl=MODULE.read_text()
            assert impl.count(old)==1
            (parent/"mutant.py").write_text(impl.replace(old,new))
            (parent/"test_wrong.py").write_text(suite)
            r=subprocess.run([sys.executable,"-m","pytest","-q",str(parent/"test_wrong.py"),"-k",test],
                             cwd=ROOT,env={**os.environ,"PYTHONPATH":os.pathsep.join([str(ROOT/"tsfm_lens"),str(ROOT)])},
                             capture_output=True,text=True)
            if r.returncode!=1 or "AssertionError" not in r.stdout or " failed" not in r.stdout:
                raise RuntimeError(f"{name}: {r.stdout}\n{r.stderr}")
            print(name, r.stdout.splitlines()[-1], flush=True)


if __name__=="__main__":
    main()
