#!/usr/bin/env bash
# Run pytest and re-emit each FAILED/ERROR summary line as a GitHub Actions
# error annotation. Job logs need an authenticated download, but annotations
# are readable from the public check-runs API, so a failure can be diagnosed
# without signing in. Usage: pytest_annotate.sh <title> <pytest args...>
title="$1"; shift
log="$(mktemp)"
python -m pytest "$@" -rfE 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
grep -E '^(FAILED|ERROR) ' "$log" | head -n 40 | while IFS= read -r line; do
  echo "::error title=${title}::${line:0:900}"
done
exit "$status"
