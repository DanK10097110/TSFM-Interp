"""Challenge the five known-answer tests in isolated temporary module copies."""
import importlib.util
import tempfile
from pathlib import Path

from research.activation_transfer.moment.crosscoder import test_evaluation as tests


def check():
    here=Path(__file__).resolve().parent
    plants=[
        ("evaluate","boundaries=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]","boundaries=np.arange(len(s))",
         "threshold","test_threshold_retains_dose_and_never_splits_tied_scores"),
        ("evaluate","eig[:,None]+relative*scale","eig[:,None]+relative",
         "calibrate","test_dual_readout_matches_independent_exhaustive_primal_validation"),
        ("evaluation_scores","ends=np.r_[np.flatnonzero(score[1:]!=score[:-1]),len(score)-1]","ends=np.arange(len(score))",
         "average_precision","test_weighted_ap_groups_ties_before_ranking"),
        ("evaluation_scores","arr[:,eligible].mean(1)","np.average(arr[:,eligible],axis=1,weights=(y[:,eligible]**2).sum(0))",
         "score","test_macro_skill_counts_false_firings_and_weights_atoms_equally"),
        ("evaluation_scores","a=(a[:,:,None]*b[:,None,:]).reshape(draws,-1)",
         "a=rng.multinomial(int(np.prod(shape)),np.full(np.prod(shape),1/np.prod(shape)),size=draws)",
         "resample","test_station_date_draws_share_one_date_weight_across_stations")]
    with tempfile.TemporaryDirectory() as directory:
        for i,(source,old,new,name,fn) in enumerate(plants):
            src=(here/(source+".py")).read_text()
            assert src.count(old)==1
            path=Path(directory)/f"plant{i}.py";path.write_text(src.replace(old,new))
            spec=importlib.util.spec_from_file_location("plant"+str(i),path)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            original=getattr(tests,name);setattr(tests,name,getattr(mod,name))
            try:
                getattr(tests,fn)()
            except AssertionError:
                print("KILLED",fn,flush=True)
            else:
                raise AssertionError("SURVIVED "+fn)
            finally:
                setattr(tests,name,original)
            getattr(tests,fn)()
            print("CLEAN",fn,flush=True)


if __name__=="__main__":
    check()
