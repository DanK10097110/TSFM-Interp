"""Induce off-by-one date and uncentered-cosine regressions in isolated scratch copies."""
import importlib.util
import tempfile
from pathlib import Path

from research.activation_transfer.moment.crosscoder.fresh_readout import test_fresh as tests, test_final_probe as ft


def run():
    base=Path(__file__).resolve().parent
    cases=[("panel","np.isfinite(v[t-480:t])","np.isfinite(v[t-479:t+1])",
            tests.test_history_eligibility_excludes_origin_day_and_counts_consecutive_missing),
           ("score","z=a-a.mean(-1,keepdims=True);q=b-b.mean(-1,keepdims=True)","z=a;q=b",
            tests.test_scope_specificity_removes_constant_level_and_uses_the_matching_readout)]
    cases.append(("final_scores",'diff=adv["probe"]-adv["native"]','diff=adv["probe"]',
                  ft.test_replacing_readout_compares_the_same_source_target_and_decoy_per_case))
    with tempfile.TemporaryDirectory() as folder:
        for source,old,new,fn in cases:
            suite=ft if source=="final_scores" else tests
            original=getattr(suite,source)
            code=(base/(source+".py")).read_text();assert code.count(old)==1
            path=Path(folder)/(source+".py");path.write_text(code.replace(old,new))
            spec=importlib.util.spec_from_file_location(source,path);mod=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod);setattr(suite,source,mod)
            try:fn()
            except AssertionError:print("KILLED",fn.__name__,flush=True)
            else:raise AssertionError("SURVIVED "+fn.__name__)
            finally:setattr(suite,source,original)
            fn();print("CLEAN",fn.__name__,flush=True)


if __name__=="__main__":
    run()
