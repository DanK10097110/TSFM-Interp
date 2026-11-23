"""Download pinned official competition files, checking hashes when a saved manifest exists."""
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION="6c1067e5a57161249b17289a565178dc7a3fb3ca"
SOURCE=Path("/tmp/tsfm_m4_calendar")


def run():
    SOURCE.mkdir(exist_ok=True,parents=True)
    for group in ("Yearly","Quarterly","Monthly"):
        manifest=Path(__file__).resolve().parent/"results"/f"{group}_manifest.json"
        expect=json.loads(manifest.read_text()) if manifest.exists() else {}
        for stage in ("train","test"):
            name=f"{group}-{stage}.csv"
            path=SOURCE/name;key=stage+"_sha256"
            def verify(p):
                with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
            if path.exists() and key in expect and verify(path)==expect[key]:continue
            url=f"https://raw.githubusercontent.com/Mcompetitions/M4-methods/{REVISION}/Dataset/{stage.title()}/{name}"
            temp=path.with_suffix(".partial")
            with urllib.request.urlopen(url,timeout=100) as response,temp.open("wb") as out:
                while block:=response.read(1<<20):out.write(block)
            digest=verify(temp)
            if key in expect and digest!=expect[key]:raise ValueError(f"{name} digest mismatch")
            temp.replace(path)
            print(name,digest,flush=True)

if __name__=="__main__":run()
