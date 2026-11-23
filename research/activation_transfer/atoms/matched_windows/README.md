# Natural-window and atom interventions under the same cross-model test

On the same 160 real daily histories, the frozen Bolt → TimesFM map reproduced the signed eight-step forecast response to a dose-matched adjacent-window direction much better than it reproduced a literal SAE atom removal. Macro cosine rose from **.074 to .898**, with paired improvement **+.824 [.767, .883]**. In the reverse direction, both full waveforms aligned fairly well, and the paired cosine improvement of +.087 had an interval spanning zero. Dose-sensitive skill improved for the window edit in both directions. [Prospective follow-up protocol](PROTOCOL.md) · [All scores, intervals and individual atoms](results/scores.json)

| Source → receiver | Measure | Atom removal | Window direction | Window − atom, paired 95% interval |
|---|---|---:|---:|---:|
| Bolt → TimesFM | Full signed cosine | .074 | .898 | +.824 [.767, .883] |
| | Horizon-centered cosine | −.121 | .199 | +.320 [.141, .441] |
| | Dose skill | −5.756 | .551 | +6.307 [4.370, 14.430] |
| | Horizon-centered dose skill | −1.607 | −.327 | +1.280 [.859, 2.373] |
| TimesFM → Bolt | Full signed cosine | .769 | .856 | +.087 [−.016, .121] |
| | Horizon-centered cosine | −.099 | .034 | +.133 [−.013, .304] |
| | Dose skill | −.007 | .404 | +.410 [.106, 1.095] |
| | Horizon-centered dose skill | −.073 | −.014 | +.059 [−.001, .144] |

Each score compares the **recipient forecast change with the source forecast change for the same intervention**. The two edit classes share a reference definition and the active feature/history cases. Each class has its own measured source response. Dose skill is `1 − sum((recipient − source)²) / sum(source²)` per feature. Zero forecast change has skill zero. We summed over firing histories and eight horizons within each source feature, then averaged equally across the 16 features. The intervals resample all 160 series jointly across features and arms, with the original 4,000 draws. These are marginal intervals conditional on the selected features and fitted map. All 16 features in both directions cleared the original source RMS floor of 0.00001 history standard deviations in both arms for both full and centered scores. [Scoring implementation](score.py)

## How the edits were matched

We reused the original atom study’s 1,024 paired clean training histories, post-block activations at layer 3 of Chronos-Bolt-small and layer 10 of TimesFM 2.5, 16 selected SAE atoms per source, feature activation amplitudes, and final 32-observation mask. This covers the last two context tokens in Bolt and the last context token in TimesFM. It produced 494 firing feature/history pairs from Bolt and 588 from TimesFM. The new arm used exactly these cases, including multiple features from the same history. [Original atom study](../README.md) · [Exact fit/evaluation IDs, external normalization and indices](results/manifest.json)

For each history we moved its 480-observation input window **backward by one observed sample**. Every series had a finite preceding observation, so the forward fallback was unused. We retained the clean history’s external mean and standard deviation, ran both original preprocessing adapters, and measured the shifted-minus-clean activation on the final 32-observation patch. For every firing source atom, we rescaled that history’s measured direction to the L2 norm of that atom’s observed removal, preserving its measured sign. A history could therefore receive the same direction at several naturally occurring doses. All 160 source directions were nonzero in both models.

The original atom edit writes `−z_f w_f` to the source and `−z_f T(w_f)` to the recipient. The matched window arm writes `d u` and `d T(u)`, where `u` is the unit-length observed adjacent-window displacement and `d` is the matched atom-removal length. The unaltered clean-map solver applies the affine map’s linear part `T` to arbitrary displacements. Both patches start with the original clean tokens, broadcast the same displacement to both Bolt tokens, and preserve the original reconstruction residual. **The window result is the forecast response to a rescaled single-block write.** Processing the shifted input itself served only to obtain its local hidden direction. [Input reconstruction and forward capture](prepare.py) · [Dose matching, map and patch implementation](patch.py)

The difference in retained full-waveform behavior persists after matching the model pair, map, dataset, atom activation cases, source hidden dose, token locations and forecast-reference definition. Differences in downstream source effect magnitude remain. From Bolt, mean per-feature full source RMS was .01174 for atom removals and .03231 for window writes. From TimesFM it was .08748 and .12646. These are model outputs in original-history standard deviations. The window arms had centered cosines of .199 [.033, .288] from Bolt and .034 [−.058, .154] from TimesFM, and both had negative centered dose skill. Much of the agreement concerns the overall forecast level. The comparison isolates these two families of directions under one calibration map, independently of whether a particular feature has a shared semantic interpretation in another model.

## Artifacts and checks

The retained arrays contain per-item IDs, feature IDs, firing masks, SAE amplitudes, source write norms, original displacement norms, mapped-vector norms and all four eight-lead effects. `atom_source` and `atom_target` reproduce the archived atom results. `window_source` and `window_target` are the newly measured single-site interventions. [494 Bolt-source cases](results/bolt_pairs.npz) · [588 TimesFM-source cases](results/timesfm_pairs.npz)

The clean histories reconstructed exactly. Maximum unshifted token replay error was 0.000000476837158203125 for Bolt and 0.00002288818359375 for TimesFM under evaluation-only re-batching. Maximum absolute difference from the archived 16 mapped decoder vectors was 4.440892098500626e−16 for Bolt source and 6.938893903907228e−18 for TimesFM source. Maximum relative per-case dose differences were below 0.000000009. Re-executed atom forecast changes differed from the archived eight-lead effects by at most 0.0000019073486328125 history SD. We used the original archived atom values for the comparison and independently checked every recomputed original macro value against the stored score. [Bolt replay](results/bolt_audit.json) · [TimesFM replay](results/timesfm_audit.json)

The original atom study’s cache supplies the cross-model training tokens and decoder-vector artifacts. Its [reproduction script](../reproduce.sh) can regenerate those inputs. The matched study then runs from the repository root with the original checkpoint and data caches available locally.

```bash
PYTHON=/common/users/djk255/conda_envs/cudaPy/bin/python \
  bash research/activation_transfer/atoms/matched_windows/reproduce.sh \
  /tmp/tsfm_atoms_20261120 /tmp/matched-window-scratch /tmp/matched-window-results
```

An independent run reconstructed all original and shifted inputs in fresh directories, recaptured both checkpoints, replayed both atom and window interventions, and recalculated the bootstrap. The maximum difference was exactly zero for each saved input, hidden displacement and eight-lead forecast component and for both point contrasts and interval limits. The three new tests passed, and all five targeted regressions produced assertion failures in fresh isolated source copies. [Numerical replay](results/replay.json) · [Mutation verification](MUTATION_CHECK.md)
