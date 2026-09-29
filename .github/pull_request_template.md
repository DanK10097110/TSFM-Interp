## What does this change and why

## Checklist

- [ ] **Tests**: new/changed behavior has a test with a planted, known
      answer (and a decoy where relevant). The plant was confirmed to make
      the test fail, then reverted to confirm it passes again — paste the
      pytest summary line for both (e.g. `2 failed in 30.4s` → `2 passed in
      31.9s`), not just "tests pass."
- [ ] **Docs checks** (if you touched a README, a stage's docstring/question
      text, the glossary, or an adapter):
  ```
  cd tsfm_lens
  python render_stage_docs.py --check
  python render_glossary.py --check
  python render_adapter_docs.py   # no diff
  ```
- [ ] **Config fingerprints**: if you added a field to a stage's config
      section, is it declared in that stage's `config_keys`
      (`tsfm_lens/manifest.py`) — or explicitly excluded with
      `metadata={"stage_input": False}` if no stage artifact depends on it?
- [ ] **Opt-in behavior**: if you added a new generator option, archetype,
      or default, does it leave every existing config's output byte-identical
      unless explicitly enabled?
- [ ] **No hardcoded absolute paths**, and all new text I/O uses
      `encoding="utf-8"`.
- [ ] Ran the relevant test suite:
  ```
  cd tsfm_benchmark && PYTHONPATH=. python -m pytest tests -q
  cd tsfm_lens && python -m pytest tests/test_smoke.py -q   # plus the stage(s) you touched
  ```

## Anything unverified or out of scope for this PR
