#!/usr/bin/env bash
# Build a self-contained source archive of the MLOSS paper (for Overleaf or the
# JMLR submission system): main.tex (all sections and the figure inline), cover_letter.tex, jmlr2e.sty and references.bib, with
# the ../ paths rewritten. The archive is test-compiled before it is zipped.
# Usage: bash make_archive.sh [output.zip]   (default: <repo root>/mloss_paper_source.zip)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$(cd "$here/../.." && pwd)/mloss_paper_source.zip}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

mkdir -p "$work"
cp "$here/main.tex" "$work/"
cp "$here/../jmlr2e.sty" "$here/../references.bib" "$work/"
cp "$here/cover_letter.tex" "$work/cover_letter.tex"
sed -i 's#{../jmlr2e}#{jmlr2e}#; s#\\bibliography{../references}#\\bibliography{references}#' "$work/main.tex"
if grep -n '\.\./' "$work/main.tex"; then
  echo "main.tex still refers to the parent directory" >&2
  exit 1
fi

(cd "$work" && pdflatex -interaction=nonstopmode main >/dev/null && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null)
if grep -qE '^!|undefined references' "$work/main.log"; then
  echo "test compilation failed; see $work/main.log" >&2
  exit 1
fi
(cd "$work" && pdflatex -interaction=nonstopmode cover_letter >/dev/null)
if grep -q '^!' "$work/cover_letter.log"; then
  echo "cover letter compilation failed; see $work/cover_letter.log" >&2
  exit 1
fi
for f in main cover_letter; do
  pdfinfo "$work/$f.pdf" | grep Pages | sed "s/^/$f /"
done
rm -f "$work"/{main,cover_letter}.{aux,bbl,blg,log,out,pdf}

rm -f "$out"
(cd "$work" && zip -qr "$out" .)
echo "wrote $out"
