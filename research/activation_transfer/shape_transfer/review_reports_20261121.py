"""Render the research reports as HTML, resolve local links and verify the recorded physical replays."""
import argparse
from collections import Counter
from hashlib import sha256
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
from urllib.parse import unquote,urlsplit

from markdown_it import MarkdownIt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SCOPE=(
    "NOV21.md","REPRODUCE_NOV21.md","../GAP.md","README.md","../README.md",
    "self_transfer","dictionary_prior","lead_budget","forecast_loss","interactions","quantiles",
    "response_subspace","depth","random_receiver","level_pair","state_predictor","feature_groups",
    "calibration_budget","rolling","method_review","blind_frequency","input_bridge",
)
REPLAYS=("replay_20261121.json","replay_20261121_followup.json","replay_20261121_final.json","replay_20261121_span.json")


class Parsed(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags=Counter()
        self.refs=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        self.tags[tag]+=1
        if tag=="a" and "href" in a:
            self.refs.append((tag,a["href"]))
        if tag=="img":
            if not a.get("alt"):
                raise ValueError("Missing descriptive image text")
            self.refs.append((tag,a["src"]))


def main(work):
    md=MarkdownIt("commonmark").enable("table")
    files=set()
    for name in SCOPE:
        p=(HERE/name).resolve()
        if p.is_dir():files.update(p.rglob("*.md"))
        else:files.add(p)
    work.mkdir(exist_ok=True,parents=True)
    review={}
    missing=[]
    linked_files=set()
    for p in sorted(files):
        html=md.render(p.read_text(encoding="utf-8"))
        parser=Parsed()
        parser.feed(html)
        if not parser.tags["h1"]:
            raise ValueError(f"Missing H1: {p}")
        local=0
        for tag,url in parser.refs:
            parsed=urlsplit(url)
            if parsed.scheme or parsed.netloc or not parsed.path: continue
            local+=1
            dest=(p.parent/unquote(parsed.path)).resolve()
            if not dest.exists():missing.append((str(p.relative_to(ROOT)),url))
            elif dest.is_file():linked_files.add(str(dest.relative_to(ROOT)))
        rel=str(p.relative_to(ROOT))
        dest=work/(rel+".html")
        dest.parent.mkdir(exist_ok=True,parents=True)
        base=escape(p.parent.as_uri()+"/",quote=True)
        style="body{max-width:920px;margin:32px auto;padding:0 18px;font:17px/1.52 system-ui;color:#202630}h1,h2,h3{line-height:1.18}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:.91em}th,td{padding:7px 9px;border:1px solid #c6cdd3}a{color:#17527f}pre{overflow:auto}"
        dest.write_text(f'<!doctype html><html><head><meta charset="utf-8"><base href="{base}"><style>{style}</style></head><body>{html}</body></html>',encoding="utf-8")
        review[rel]=dict(source_sha256=sha256(p.read_bytes()).hexdigest(),links=len(parser.refs),
                         local=local,tables=parser.tags["table"],images=parser.tags["img"])
    if missing:raise ValueError(json.dumps(missing,indent=2))
    visible=subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard","-z"],cwd=ROOT)
    excluded=linked_files-set(visible.decode("utf-8").split("\0"))
    if excluded:raise ValueError(f"Referenced files excluded by Git: {sorted(excluded)}")
    checked=zero=arrays=0
    for file in REPLAYS:
        with (HERE/file).open() as f: source=json.load(f)
        for job,archives in source.items():
            if job=="copied_sources":continue
            checked+=1
            for record in archives.values():
                if record["identical_values"]:zero+=1
                else:raise ValueError(f"Non-identical output in {job}")
                arrays+=record["arrays"]
    summary=dict(reports=len(review),local_references=sum(v["local"] for v in review.values()),
                 tables=sum(v["tables"] for v in review.values()),images=sum(v["images"] for v in review.values()),
                 missing_local_paths=[],git_excluded_local_files=[],distinct_linked_files=len(linked_files),
                 exact_replay_jobs=checked,exact_replay_archives=zero,
                 exact_replay_arrays=arrays,reports_checked=review)
    (HERE/"validation_20261121.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print({k:v for k,v in summary.items() if k!="reports_checked"},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--render",type=Path,default=Path("/tmp/tsfm_nov21_render"))
    main(p.parse_args().render)
