"""Run targeted numerical regression plants on isolated copies of the lookup implementation."""
from pathlib import Path
import subprocess
import sys
import tempfile

src=Path(__file__).resolve().parent
cases=[
    ("fit","Symmetric difference", "test_support_neighbor_ties_and_order_match_symmetric_difference",
     "qnorm[:,None]+tnorm[None,:]-2*cross","qnorm[:,None]+tnorm[None,:]-cross"),
    ("fit","Validation shrinkage","test_shrinkage_uses_known_generic_validation_truth_and_clamps_negative_signal",
     "np.clip(correlation/variance,0,1)","np.clip(variance/correlation,0,1)"),
    ("fit","Generic response shrinkage","test_generic_neighbor_average_predicts_new_decoder_columns_linearly",
     'settings["mix"]*(memory[idx].mean(1)-avg)','(memory[idx].mean(1)-avg)'),
    ("ties","Stable tie-breaking","test_fractional_boundary_shares_remaining_mass_and_is_permutation_invariant",
     "weight=(near+at*(remainder/at.sum(1))[:,None])/k","weight=np.zeros_like(dist); np.put_along_axis(weight,np.argsort(dist,axis=1,kind='stable')[:,:k],1./k,axis=1)"),
]
with tempfile.TemporaryDirectory(prefix="state_predictor_plants_") as tmp:
    for i,(module,name,test,old,new) in enumerate(cases):
        directory=Path(tmp)/str(i);directory.mkdir()
        source=(src/(module+".py")).read_text()
        assert source.count(old)==1
        (directory/"mutant.py").write_text(source.replace(old,new))
        suite=(src/"test_predictor.py").read_text().replace("research.activation_transfer.shape_transfer.state_predictor."+module+" import","mutant import")
        (directory/"test_wrong.py").write_text(suite)
        r=subprocess.run([sys.executable,"-m","pytest","-q",str(directory/"test_wrong.py")+"::"+test],
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if r.returncode!=1 or "AssertionError" not in r.stdout or "1 failed" not in r.stdout:
            raise ValueError(name+" escaped or failed outside the numerical assertion: "+r.stdout)
        print(name,r.stdout.splitlines()[-1],flush=True)
