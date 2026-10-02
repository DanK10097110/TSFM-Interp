# Paper: TSFM-Lens (JMLR drafts)

Two versions: the full JMLR paper (`main.tex`, under 30 pages including references and appendix)
and a short JMLR MLOSS-track software paper (`mloss/main.tex`, under 4 pages excluding references,
sharing `jmlr2e.sty` and `references.bib`; build it from inside `mloss/` with the same commands).

`main.tex` + `sections/*.tex` + `references.bib`, using the official JMLR style
(`jmlr2e.sty`, from https://github.com/JmlrOrg/jmlr-style-file). Build:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Every data figure is regenerated from the published example run:

```bash
python figures/make_figures.py            # reads ../examples/panel7_v2/run (--run ../examples/concept_atlas_v2/run for the four-model run)
python figures/make_figures.py --run <any run dir>
```

`figures/report_*.png` are headless-Chrome screenshots of (`report_l3_crop.png` is a crop of `report_sec-l3.png`)
`examples/concept_atlas_v2/report.html` (the four-model run); the case-study numbers are from `examples/panel7_v2`.

Before submission: remove the `preprint` option, add author affiliations, and spot-check bibliography
metadata (entries marked `verified` in `references.bib` were checked against
arXiv/venue pages on 2026-09-29).
