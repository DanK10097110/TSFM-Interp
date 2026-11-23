"""Planted defects against independent numerical assertions; never change working code."""
from pathlib import Path
from subprocess import run, PIPE, STDOUT
import sys
import tempfile

source=Path(__file__).resolve().parent
plants=[
    ("probe","test_public_input_directions_have_known_decoder_and_touch_only_final_32","sample = np.arange(WINDOW) + .5","sample = np.arange(WINDOW) + 1."),
    ("probe","test_public_input_directions_have_known_decoder_and_touch_only_final_32","out[..., -WINDOW:] +=","out[..., :WINDOW] +="),
    ("run","test_weighted_minimum_uses_each_history_and_correct_source_goal_for_both_models","ridge = .001 * s[:, :1]**2","ridge = .001 * s[:1, :1]**2"),
    ("run","test_normalized_tail_budget_and_scramble_preserve_physical_size","return weights * scale[..., None], norms","return weights, norms"),
    ("run","test_normalized_tail_budget_and_scramble_preserve_physical_size","return np.roll(coeff, 5, axis=-1) * sign","return coeff * sign"),
]
with tempfile.TemporaryDirectory(prefix="tsfm_input_bridge_defects_") as temp:
    for i,(module,test,old,new) in enumerate(plants):
        folder=Path(temp)/str(i);folder.mkdir()
        code=(source/(module+".py")).read_text()
        assert code.count(old)==1
        (folder/"mutant.py").write_text(code.replace(old,new))
        suite=(source/"test_bridge.py").read_text().replace("research.activation_transfer.shape_transfer.input_bridge."+module+" import","mutant import")
        (folder/"test_plant.py").write_text(suite)
        res=run([sys.executable,"-m","pytest","-q",str(folder/"test_plant.py")+"::"+test],stdout=PIPE,stderr=STDOUT,text=True)
        if res.returncode!=1 or "AssertionError" not in res.stdout or "1 failed" not in res.stdout:
            raise ValueError("Plant escaped or failed unexpectedly: "+res.stdout)
        print(module,test,res.stdout.splitlines()[-1],flush=True)
