#!/usr/bin/env bash
# Build a self-contained source archive of the findings paper (for Overleaf or the
# JMLR submission system): main.tex, jmlr2e.sty, references.bib and the figures, with
# the ../ paths rewritten. The archive is test-compiled before it is zipped.
# Usage: bash make_archive.sh [output.zip]   (default: <repo root>/findings_paper_source.zip)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$(cd "$here/../.." && pwd)/findings_paper_source.zip}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/figures"
cp "$here/main.tex" "$work/"
cp "$here/../jmlr2e.sty" "$here/../references.bib" "$work/"
cp "$here"/figures/*.pdf "$work/figures/"
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
rm -f "$work"/main.{aux,bbl,blg,log,out,pdf}

rm -f "$out"
(cd "$work" && zip -qr "$out" .)
echo "wrote $out"
