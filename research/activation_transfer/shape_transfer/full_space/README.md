# A history-specific translation of genuine SAE removals

On 192 held-out M4 daily histories, the receiving checkpoint reproduced **.859** of the horizon-centered source effect for TimesFM → Bolt and **.944** for Bolt → TimesFM. The respective 95% paired series-bootstrap intervals were [.842, .887] and [.925, .974]. The centered outcome subtracts each eight-lead change's own mean before scoring. A method that only raises or lowers the whole forecast gets zero. Each of the 14 and 15 eligible source atoms contributes equal weight. Including all 16 selected atoms gives .824 and .947.

We removed a firing feature's actual decoder contribution from the source's live state, preserving the reconstruction residual. To translate that intervention, we measured how each model's eight outputs respond to small writes at its own SAE site on the **current unedited history**. Multiplying the source derivative by the proposed removal predicts an eight-lead change. We solved for a target write that would produce that change, then applied it to the actual frozen target. The solver sees ordinary-state derivatives and the natural source coefficient. The source's forecast after removal is reserved for evaluation. Eight reverse passes per checkpoint on a batch give derivatives reusable across all of its selected features. The target translation can use its full hidden width. [Protocol and controls](PROTOCOL.md) · [Derivative measurement](jacobians.py) · [Live writes](transfer.py)

| Recovery of centered forecast change | TimesFM → Bolt, daily | Bolt → TimesFM, daily | TimesFM → Bolt, VMs | Bolt → TimesFM, VMs |
|---|---:|---:|---:|---:|
| Previous clean-state map | −.114 | −1.240 | −.059 | −.880 |
| Fixed map fitted to derivatives on 96 calibration histories | .444 | .238 | −.107 | −.105 |
| Adaptive recipient, donor derivative averaged on calibration | .479 | .393 | .208 | −.976 |
| Both models' derivatives on current history | **.859** | **.944** | **.568** | **.348** |
| Same procedure, wrong lead order | −.423 | −1.092 | −.157 | −1.522 |
| Forecast-only template fitted to true calibration ablations of each feature | .497 | .395 | .447 | −.512 |
| Donor's own first-order approximation, without running a target | .902 | .945 | .765 | .915 |
| Frequent source features / active feature–history pairs | 14 / 691 | 15 / 520 | 2 / 156 | 4 / 127 |

Scores are one minus squared error divided by the energy of the true source change, computed per feature over its firing histories and leads, then averaged. Negative values can be arbitrarily large in magnitude. Centered shape accounted for 4.3% and 1.9% of these features' full source-effect energy on daily histories. Even the wrong-order control had full-waveform skills of .867 and .955 on daily histories. Removing the forecast level exposed that control's error. Its norm cap matched the genuine solver's cap, but achieved write lengths could differ. The 128 VM histories had few frequently firing features and a wide interval for Bolt → TimesFM [.205, .980]. [All arms, intervals and per-feature values](results/results.json) · [Prior feature selection](../../replication/README.md)

### Can the recipient be calibrated once?

Yes on daily histories. We fitted **eight reusable receiver write vectors** on 96 calibration histories, one for each unit forecast lead. A linear combination of those vectors can implement a donor's predicted response on a new history with no new receiver derivative. With the donor's current response and this fixed receiver control, centered recovery reached **.796** [.770, .816] into Bolt and **.896** [.859, .934] into TimesFM. With the donor's calibration-average derivative and the same fixed receiver it fell to .434 and .331. With both models measured on each history it was .859 and .944. Most of the daily benefit came from asking how the *donor's feature* would affect this series. All arms used the current donor coefficient.

On the machines, the same fixed receiver obtained .203 and .612 versus .568 and .348 with both local derivatives. One strongly capped direction benefited from adapting the receiver and the other direction's fixed receiver had better full-dose centered skill. The dose experiments below explain why an inverse that is accurate infinitesimally can perform worse for a finite action. [Access-controlled protocol](ACCESS_PROTOCOL.md) · [Two-by-two results](results/followups.json)

To reuse the calibrated controller, take an eight-vector `g` in **signed forecast-change units**, form `receiver_library @ g`, and radially cap its length at `0.20 * radius`. Add that same vector at the recorded post-block token positions for the final 32 observations. The archived study uses the current source coefficient in `g` exactly once. The files are named for the donor and each records its receiver checkpoint family, layer, token indices and calibration IDs. [Bolt-source library for TimesFM](results/bolt_access.npz) · [TimesFM-source library for Bolt](results/timesfm_access.npz)

### Does it transfer beyond the leads it can inspect?

| Centered daily recovery | TimesFM → Bolt | Bolt → TimesFM |
|---|---:|---:|
| Constrain leads 1, 3, 5, 7; score leads 2, 4, 6, 8 | .572 | .714 |
| Interpolate the same four donor predictions, without editing a target | .621 | .715 |
| Constrain all first eight leads; score leads 9–16 | −.215 | −.287 |

The intervention still produced related responses on neighboring withheld leads. The forecast-only interpolation matched or exceeded those point estimates. On daily histories neither transfer direction preserved the later *shape* at its original dose. On VMs the corresponding late shape scores were positive, .348 and .144, across two and four frequent atoms. The algorithm never used derivatives from leads 9–16. These are examples of agreement over a specified set of output leads and mixed evidence for a more general temporal correspondence. [Withheld-lead protocol](LEADS_PROTOCOL.md) · [Full and centered scores on exposed, omitted and future leads](results/followups.json)

### Why did the VM result fall?

* **TimesFM → Bolt.** The target write hit its 20%-radius cap on 136 of 156 cases. Doubling the cap reduced this to 109 and increased centered recovery from .568 to .664. The cap accounts for part of the deficit. The donor's own derivative predicted its natural ablation with .765 recovery, already below the daily .902.
* **Bolt → TimesFM.** None of 127 target writes hit the cap. The recipient's derivative predicted .915 recovery but its actual forecast yielded .348. Scaling the original target write and the source removal to one quarter of their original strengths yielded .963 against the *actual quarter-dose source ablation*. At that dose the target linear prediction scored .979. Finite-dose curvature in the target forecast is a plausible cause of the original loss here, even with a small relative hidden displacement. The quarter-dose task has its own reference effect.

All new fits preserve the original split and physical intervention positions. A generic inverse can realize a source forecast response using a different target mechanism. The practical costs are white-box access to both checkpoints, per-history differentiation and a known set of corresponding forecast leads. The local method provides no fixed target feature ID. [Dose intervention and paired uncertainty](DOSE_PROTOCOL.md) · [Dose and late-lead records](results/followups.json) · [Translation using the receiver's own SAE](../sparse_writes/README.md)
