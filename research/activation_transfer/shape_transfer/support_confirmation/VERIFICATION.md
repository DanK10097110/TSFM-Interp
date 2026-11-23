# Verification

| Check | Result |
|---|---|
| Six new M4 study tests | Passed. Eight planted regressions caused the designated assertion failures. |
| Three external station-block tests | Passed. Each of three planted implementation errors caused its designated assertion failure. |
| Combined package tests and lens smoke suite | 14 passed. The pre-existing end-to-end smoke test produced a PytestReturnNotNoneWarning. |
| Prospective M4 physical replay | 180 independent singleton interventions, maximum absolute difference 0.000003814697265625 observed-history SD. |
| Projection and orthogonalization replay | 21 independent singleton interventions, maximum absolute difference 0.0000011920928955078125 SD. |
| Derivative/support factorial replay | 60 independent singleton interventions, maximum absolute difference 0.000003814697265625 SD. |
| Isolated source tree | Reconstructed all three JSON replay files byte-for-byte from copied source and separate copies of measured outcome archives. |
| External weather physical replay | 64 independently reconstructed singleton edits, maximum difference 0.000001433 historical SD. The isolated source and artifact copy reproduced the replay byte-for-byte. |
| Weather report | Rendered with both tables and every figure and source link checked. Reviewed the station-block interval chart. |
| M4 report | Rendered the three results documents to HTML, checked each image and link target and verified that each table row had the same number of cells as its header. Inspected the scientific comparison figure. |

[Primary replay data](results/replay.json) · [Residual replay data](results/residual_replay.json) · [Derivative replay data](results/derivative_replay.json) · [Isolated source replay](replay_isolated.sh) · [M4 mutation record](MUTATION_CHECK.md) · [Station mutation record](noaa/MUTATION_CHECK.md)
