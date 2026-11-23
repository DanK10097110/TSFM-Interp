"""Test the depth-pairing and deployment information barriers using isolated regressions."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
here=Path(__file__).resolve().parent;root=here.parents[4]
for old,new,name in [('rows if index<2 else perm[rows]','rows if index<3 else perm[rows]','vertical_derangement_preserves'),
                     ('n=1 if source_only else len(NAMES)','n=2 if source_only else len(NAMES)','source_pass_suppresses')]:
    with tempfile.TemporaryDirectory(prefix='multiview_cc_plant') as td:
        dest=Path(td)/here.relative_to(root);dest.mkdir(parents=True)
        for file in ('train.py','test_multilayer_training.py'):shutil.copy(here/file,dest/file)
        file=dest/'train.py';src=file.read_text();assert src.count(old)==1;file.write_text(src.replace(old,new))
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([td,str(root),str(root/'tsfm_lens'),env.get('PYTHONPATH','')])
        p=subprocess.run([sys.executable,'-m','pytest','-q',str(dest/'test_multilayer_training.py'),'-k',name],
                       cwd=td,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        print(name,p.stdout.splitlines()[-1],flush=True)
        assert p.returncode==1 and '1 failed' in p.stdout and name in p.stdout,p.stdout
