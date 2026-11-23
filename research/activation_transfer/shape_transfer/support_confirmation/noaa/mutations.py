"""Probe station-block test sensitivity on isolated source copies."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

from research.activation_transfer.shape_transfer.support_confirmation.data import ROOT
from research.activation_transfer.shape_transfer.support_confirmation.noaa.data import HERE

PLANTS=[
    ('station_eligibility','score.py','.any(0)','.sum(0)','Treat multiple origins of one station as different eligible stations.'),
    ('station_derangement','run.py','lookup[donor[s],o]','lookup[donor[s],(o+1)%4]','Compare the next season at the other station.'),
    ('station_equal','score.py','if equal_stations:','if False:','Drop the inverse number-of-firings weight for equal active stations.'),
]


if __name__=='__main__':
    result=[]
    for key,file,a,b,reason in PLANTS:
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);package=folder/HERE.relative_to(ROOT);package.mkdir(parents=True)
            for p in HERE.glob('*.py'):shutil.copy2(p,package/p.name)
            src=package/file;s=src.read_text(encoding='utf-8');assert s.count(a)==1;src.write_text(s.replace(a,b),encoding='utf-8')
            env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[folder,ROOT,ROOT/'tsfm_lens'])))
            proc=subprocess.run([sys.executable,'-m','pytest','-q',str(package/'test_weather_transfer.py'),'-k',key],cwd=folder,env=env,text=True,capture_output=True)
            assert proc.returncode==1 and '1 failed' in proc.stdout and 'AssertionError' in proc.stdout,proc.stdout
            result.append(f'| {key} | {reason} | {proc.stdout.strip().splitlines()[-1]} |')
    (HERE/'MUTATION_CHECK.md').write_text('# Station inference test mutations\n\nThree tests passed on the original source. Each planted implementation error below failed its expected assertion in an isolated copy.\n\n| Test | Plant | Result |\n|---|---|---|\n'+'\n'.join(result)+'\n',encoding='utf-8')
