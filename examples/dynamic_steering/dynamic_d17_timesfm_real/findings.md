# Real-context audit: profile migration failure

The saved early profile used JSON integer `period=24`; the later request
normalizer canonicalized it to float `24.0`. Strict string key comparison
rejected the profile before evaluation, despite equal request semantics.
No real-context forecast result was obtained. The loader now compares parsed
request semantics and migrates to the canonical key. Preserve this run and
retry under a new output directory.
