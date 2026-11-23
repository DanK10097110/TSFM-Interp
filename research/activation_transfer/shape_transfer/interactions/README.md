# Does transferring two features separately preserve their interaction?

Usually the first eight forecasts moved in roughly the right way when the two transferred edits were applied together. That score concealed a harder failure. The target generally missed the small *extra* effect caused by removing both donor features together. [Prospective factorial design](PROTOCOL.md)

We ran all four conditions in the actual source and receiver: neither removal, A only, B only, both. Define interaction as the double-removal change minus the sum of the two single-removal changes. It answers a concrete question: does removing A change what happens when B is removed? All three patched arms share the same zero patch and target dose scaling. The receiver obtains separate ordinary-state derivatives for A and B. It receives no combined donor response. We include every pair among the frozen 16 source features that fired together on at least eight distinct histories in a domain.

| Additional daily histories | Qualifying pairs | Joint shape, real receiver | Joint shape, sum of two measured source singletons | Interaction alone, real receiver | Interaction alone, donor as its own translated receiver |
|---|---:|---:|---:|---:|---:|
| Bolt → TimesFM | 16 | .950 | .992 | −.010 [−.042, .016] | .170 |
| TimesFM → Bolt | 25 | .815 | .968 | −.046 [−.105, −.014] | .011 |

The two original daily directions gave joint shape .959 and .793 and interaction skill −.003 and −.055. On the additional daily set the pooled centered interaction RMS was 7.3% and 14.1% of the source singleton-sum RMS. For the Bolt donor, none of the 339 *singleton* cap decisions in the earlier daily study, or any qualifying A/B factorial write in this cohort, required shrinking the receiving edit. In the reverse direction 168 of 475 qualifying pair observations were capped. In the latter case, recipient amplitude can contribute to a difference in interactions. A pair-specific common scalar prevents clipping from creating an apparent interaction in a linear receiver. The score refits no directions or doses to a joint donor outcome. [Full cohort and unclipped-stratum scores](results/summary.json)

The ordinary-versus-zero-patch numerical discrepancy makes little difference. Using a consistently patched baseline changed no cross-model early-shape interaction estimate by more than .000231. Removing all pair observations incident on a single donor atom gave interaction-skill ranges of −.018 to .001 and −.061 to −.021 on the additional daily cohort. The original VM and traffic results have far fewer recurring pairs. TimesFM had just one qualifying pair in each domain, and the large negative Bolt → TimesFM VM result was highly dependent on which of four pairs were included. [Independent review of pair dependence and numerical resolution](../method_review/REVIEW.md)

Strong agreement about an additive sum gives weak evidence about the shared nonlinear behavior of two features. The follow-up asks how well a deliberately *pair-dependent* edit can recover the interaction when supplied four partial-dose donor measurements. [Partial-dose confirmation protocol](CONDITIONAL.md)

## Can the interaction transfer if we actually measure its dependence on the pair?

We kept the full-dose Bolt pair effect out of the controller. We measured four *partial* simultaneous removals on the donor at (¼,¼), (¼,¾), (¾,¼), and (¾,¾) of the two features' natural doses. Four times their signed mixed difference estimates the average interaction over the full dose square when the mixed curvature is reasonably uniform. The controller asked TimesFM for that predicted early interaction, correcting the original combined write by the difference between this request and the receiver's own measured interaction. The singleton writes stayed fixed. The new recipient joint edit is therefore explicitly conditional on the pair. No eligible write in these three daily sets hit the target cap. [Partial-dose specification and alternatives](CONDITIONAL.md) · [Actual four-corner probes and correction](conditional.py)

| Bolt → TimesFM cohort | Eligible pairs / cofiring cases | Original interaction | Donor four-probe estimate | Corrected physical interaction | Permuted-lead correction | Corrected late interaction |
|---|---:|---:|---:|---:|---:|---:|
| Original 192 daily | 21 / 489 | −.004 | .399 | .405 | −2.149 | −.115 |
| Additional 128 daily | 16 / 278 | −.011 | .428 | .426 | −1.881 | −.428 |
| Third 128 daily | 19 / 303 | .008 | .466 | .461 | −2.039 | −.290 |

The third cohort's physical early interaction score was .461 [.230, .559], an improvement of .453 [.224, .558] over independent translation. Asking for zero gave .000. Using only the lower-left quarter of the donor dose square estimated the interaction at .067 on that cohort. Curvature sampled around the interior generalized better to the full-dose square. Joint skill moved little (.970 to .972), consistent with the interaction being a small part of the combined forecast. The receiver saw only the eight early partial-dose coordinates. The stored donor-only *late* partial-dose estimates read the corresponding late outputs and therefore have privileged access for that diagnostic. Resampling whole histories leaves pair identities and their overlapping features fixed. A designed pair-specific write can recover part of an interaction once it has been characterized in the source. Separate writes did not predict this interaction on their own. [All three cohorts, controls, uncertainty and measured RMS](results/conditional.json)
