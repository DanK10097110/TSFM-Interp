# Can a single calibration history replace the larger library?

Once four histories showed surprisingly strong original TimesFM → Bolt performance, we registered three additional 48–96-call comparisons on the separate validation cohorts before inspecting any fresh budget score. The single-history column averages eight separately fitted and applied one-history libraries. The matched 3 × 8 and 2 × 24 candidates use the three original permutations. All singles have different calibration series IDs. We did not choose the best history or ensemble the edits. [Follow-up protocol](SINGLE_HISTORY_PROTOCOL.md) · [Physical interventions](single_history.py)

| Direction | Separate cohort | Full 96 × 24 centered skill | 1 × 24 mean (range across 8 histories) | 3 × 8 | 2 × 24 |
|---|---|---:|---:|---:|---:|
| Bolt → TimesFM | M4 daily | 0.752 | 0.227 (-0.045 to 0.392) | -0.015 | 0.285 |
| Bolt → TimesFM | Hourly traffic | 0.158 | -1.427 (-4.347 to 0.042) | -0.887 | -2.794 |
| TimesFM → Bolt | M4 daily | 0.583 | 0.477 (0.278 to 0.561) | 0.142 | 0.531 |
| TimesFM → Bolt | Hourly traffic | 0.455 | 0.477 (0.358 to 0.562) | 0.175 | 0.490 |

On fresh daily TimesFM → Bolt, the two-history design lost .052 centered skill relative to full calibration (paired 95% interval −.098 to −.005). It narrowly meets the study’s .10 tolerance for these three choices of calibration histories. The eight single-history libraries lost .106 on average (−.142 to −.070), and the individual fresh daily scores ranged from .278 to .561. On traffic their mean .477 was close to the full .455. In the reverse transfer direction a single history frequently produced severe errors, particularly on traffic, and two histories still performed poorly. The 48-call matched comparison of 1 × 24 versus 3 × 8 favors breadth on the daily domains; on Bolt → TimesFM traffic the comparison favors the narrow span because its errors are less severe. [Per-fit outcomes and paired uncertainty](results/single_scores.json) · [Raw Bolt donor cases](results/bolt_single.npz) · [Raw TimesFM donor cases](results/timesfm_single.npz)

The striking negative Bolt → TimesFM traffic skill did not generally reflect a reversed sign. Across the five frequent features and eight one-history fits, the signed horizon-centered cosine averaged .617. Hidden writes produced 3.764 times the donor’s centered forecast energy on average, with projected gain 1.168. The identity `skill = 2 × gain − relative energy` gives −1.427. Four of five features had negative mean skill across calibration choices. These effects were measured after the shared hidden-norm cap. Rescaling output curves after observing the reference was never used to score these interventions. [Feature-level amplitude breakdown](results/amplitude.json)
