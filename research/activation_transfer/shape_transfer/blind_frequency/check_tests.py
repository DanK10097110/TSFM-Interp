"""Run targeted regression plants in isolated temporary source copies."""
from pathlib import Path
import subprocess, sys, tempfile
src=Path(__file__).resolve().parent
workspace=tempfile.TemporaryDirectory(prefix="tsfm_blind_freq_mutants_")
out=Path(workspace.name)
cases=[("geometry","freq[:,3:]=0","freq[:,4:]=0","test_blind_mask_keeps_the_early_horizon_and_removes_only_fast_modes"),
       ("geometry","coeff,current-mean_e","coeff,current","test_own_history_regression_restores_channel_means_and_temporal_orientation"),
       ("geometry","np.roll(order,1)","np.roll(order,0)","test_controls_derange_entire_distinct_histories"),
       ("score","collapse(boot,feature)","collapse(boot,np.ones_like(feature))","test_crossed_bootstrap_draws_the_same_features_across_all_histories")]
for module,old,new,test in cases:
    d=out/test;d.mkdir(exist_ok=True,parents=True)
    content=(src/(module+".py")).read_text()
    assert content.count(old)==1
    (d/"mutant.py").write_text(content.replace(old,new))
    suite=(src/"test_blind.py").read_text().replace("research.activation_transfer.shape_transfer.blind_frequency."+module+" import","mutant import")
    (d/"test_wrong.py").write_text(suite)
    run=subprocess.run([sys.executable,"-m","pytest","-q",str(d/"test_wrong.py")+"::"+test],
                       text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(test,run.stdout.splitlines()[-1],flush=True)
    (d/"output.txt").write_text(run.stdout)
    if run.returncode!=1 or "1 failed" not in run.stdout or "AssertionError" not in run.stdout:raise ValueError(run.stdout)
