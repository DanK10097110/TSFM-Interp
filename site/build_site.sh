#!/usr/bin/env bash
# Assemble the GitHub Pages site into $1 (default: site_out) from files already
# in the repo. Missing optional files are skipped with a message, and their
# entries are removed from the landing page so no link is dead.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-site_out}"
rm -rf "$OUT"
mkdir -p "$OUT"
cp site/index.html "$OUT/index.html"
touch "$OUT/.nojekyll"
for f in examples/panel7_v2/report.html examples/concept_atlas_v2/report.html \
         TSFM-Lens_JMLR.pdf TSFM-Lens_MLOSS.pdf; do
  if [ -f "$f" ]; then
    mkdir -p "$OUT/$(dirname "$f")"
    cp "$f" "$OUT/$f"
    echo "included: $f"
  else
    echo "skipped (not in this checkout): $f"
  fi
done
python3 - "$OUT" <<'PY'
import re, sys
from pathlib import Path
out = Path(sys.argv[1])
p = out / "index.html"
html = p.read_text(encoding="utf-8")
def keep(m):
    return m.group(0) if (out / m.group(1)).is_file() else ""
html = re.sub(r'<li data-requires="([^"]+)">.*?</li>\n?', keep, html, flags=re.S)
html = re.sub(r'<h2>[^<]*</h2>\n<ul>\s*</ul>\n?', "", html)
html = re.sub(r' data-requires="[^"]*"', "", html)
p.write_text(html, encoding="utf-8")
PY
echo "site assembled in $OUT"
