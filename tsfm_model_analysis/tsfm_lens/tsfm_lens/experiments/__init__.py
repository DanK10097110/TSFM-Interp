"""One-off research scripts that are not pipeline stages.

`finetune_child.py` is the first entry (`ROADMAP.md` sec 23.3 D2 / sec
6.3.1 Option E): builds a genuine known-lineage pair for provenance
detection, since the repo's only prior "positive" example
(chronos-t5-small vs chronos-t5-base) turned out not to share
initialization weights at all.
"""
