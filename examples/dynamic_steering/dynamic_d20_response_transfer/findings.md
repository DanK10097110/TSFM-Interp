# Response transfer audit

{
  "n": 12,
  "successes": 8,
  "search_calls": 0,
  "mean_error": 0.17004576459792609,
  "replay_max_abs": 0.0,
  "model_calls": 396,
  "seconds": 40.386829502065666
}

Only the level profile was retained; new uncertainty/seasonal and negative-level requests reuse its measured response. No new site search was called. Selection is correctly labeled online; local response refinement remains allowed. This is response reuse, not held-out validation of the new requests or a universal fixed vector.
