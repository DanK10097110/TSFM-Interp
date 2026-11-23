"""Copy regenerable numerical arrays and fitted dictionaries into a local artifact directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from research.activation_transfer.boundary.capture import digest


def archive(source, destination, manifest):
    destination.mkdir(exist_ok=True,parents=True)
    files = sorted(list(source.glob('*.npz'))+list(source.glob('seed*_*.pt'))+list(source.glob('seed*_*.json')))
    entries = {}
    for path in files:
        target = destination/path.name
        if path.name.endswith('.tmp.npz'):
            continue
        expected = digest(path)
        if not target.exists() or digest(target) != expected:
            temp=target.with_suffix(target.suffix+'.partial')
            shutil.copyfile(path,temp)
            if digest(temp)!=expected:
                raise OSError('checksum mismatch while copying '+path.name)
            temp.replace(target)
        entries[path.name]={'sha256':expected,'bytes':path.stat().st_size}
    manifest.write_text(json.dumps(entries,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    args=p.parse_args()
    archive(args.source,args.destination,args.manifest)
