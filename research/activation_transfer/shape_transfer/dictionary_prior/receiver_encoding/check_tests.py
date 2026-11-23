"""Mutation-check load-bearing numerical assertions on isolated copies."""
from pathlib import Path
import subprocess
import sys
import tempfile

root=Path(__file__).resolve().parent
cases=[
    ("run","test_crossing_keeps_true_endpoint_boolean_counts_and_distinct_decoder_writes",
     'return ({k:variants[k] for k in ("hard","locked","shoulder")}', 'return ({k:variants["hard"] for k in ("hard","locked","shoulder")}'),
    ("run","test_projection_of_known_vector_into_dependent_decoder_span_and_physical_cap",
     'return (v@use.T)@use','return v.copy()'),
    ("run","test_projection_of_known_vector_into_dependent_decoder_span_and_physical_cap",
     'return v*np.minimum(1.,limit/np.maximum(n,1e-16))[:,None], n','return v, n'),
    ("matched","test_equal_gate_length_uses_the_actual_dense_length_and_preserves_orientation_or_zero",
     'lengths=np.minimum(desired,absolute_cap)', 'lengths=np.minimum(n,absolute_cap)'),
]
with tempfile.TemporaryDirectory(prefix="receiver_sae_regressions_") as tmp:
    for n,(module,test,old,new) in enumerate(cases):
        p=Path(tmp)/str(n);p.mkdir()
        impl=(root/(module+".py")).read_text()
        assert impl.count(old)==1
        (p/"mutated.py").write_text(impl.replace(old,new))
        (p/"test_plant.py").write_text((root/"test_encoding.py").read_text().replace("research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding."+module+" import","mutated import"))
        r=subprocess.run([sys.executable,"-m","pytest","-q",str(p/"test_plant.py")+"::"+test],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if r.returncode!=1 or "AssertionError" not in r.stdout or "1 failed" not in r.stdout:
            raise ValueError(r.stdout)
        print(test,r.stdout.splitlines()[-1],flush=True)
