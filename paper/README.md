# Paper: TSFM-Lens (JMLR drafts)

Two versions: the full JMLR paper (`main.tex`, a 32-page body plus appendices and references, 43 pages in total;
JMLR sets no page limit for regular papers)
and a short JMLR MLOSS-track software paper (`mloss/main.tex`, under 4 pages excluding references,
sharing `jmlr2e.sty` and `references.bib`; build it from inside `mloss/` with the same commands).
A third, findings-first version (`findings/main.tex`, a single .tex file, 24 pages in total) reorganizes the same results
by level of comparison (agreement on inputs, rare agreement on effects, geometry in between, with an input control).
Its figures are drawn at the 6.0 in text width by `findings/figures/make_figures.py` (from `examples/panel7_v2/run`, and
`findings/figures/data/input_control.json`, the output of `tsfm_lens/run_input_control_cka.py`, for the geometry figure;
`findings/figures/data/exemplars.json`, extracted from stored ablation forecasts with `--extract-exemplars-from tsfm_lens/runs/panel7_v2_dev`, for the concept example)
so that every label prints at 8 pt or larger. Build it from inside `findings/`;
`bash findings/make_archive.sh` writes a self-contained, test-compiled source archive (Overleaf or JMLR upload) to the
repository root. Its cover letter is `findings/cover_letter.tex` (build with pdflatex).

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
