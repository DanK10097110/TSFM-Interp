# Paper: TSFM-Lens (JMLR draft)

`main.tex` + `sections/*.tex` + `references.bib`, using the official JMLR style
(`jmlr2e.sty`, from https://github.com/JmlrOrg/jmlr-style-file). Build:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Every data figure is regenerated from the published example run:

```bash
python figures/make_figures.py            # reads ../examples/concept_atlas_v2/run
python figures/make_figures.py --run <any run dir>
```

`figures/report_*.png` are headless-Chrome screenshots of
`examples/concept_atlas_v2/report.html`.

Before submission: remove the `preprint` option, fill in the funding /
competing-interests statement in `main.tex`, and spot-check bibliography
metadata (entries marked `verified` in `references.bib` were checked against
arXiv/venue pages on 2026-09-29).
