# Where should the receiving model be steered?

Moving a write changed two different things. The late TimesFM blocks made the first eight forecast values especially easy to control using a *fixed* library of eight receiver vectors. The same final-block write had the poorest continuation on daily series. In Bolt the corresponding fixed library showed little systematic improvement with depth. A universal best relative layer is a poor description of these checkpoints. [Registered depth sweep](PROTOCOL.md) · [All sites, conditional intervals and doses](results/summary.json)

The source stayed fixed throughout. A real donor SAE feature supplied its natural removal dose and its estimated effect on the first eight forecast leads for that history. We changed only the receiving block. Each actual receiver write acted on the block output for the last 32 observed time steps. The original 20% limit was expressed relative to that block's own typical hidden length. A fixed library used only 96 separate calibration histories. An adaptive write also used the current receiving Jacobian. Five blocks per architecture were selected before measuring effects.

| Recipient / method, 128 additional M4 daily | First | Quarter | Middle | Three quarters | Final |
|---|---:|---:|---:|---:|---:|
| TimesFM, fixed, first-eight shape | .890 | .914 | .920 | .930 | **.972** |
| TimesFM, adaptive, first-eight shape | .970 | .971 | .971 | .971 | .972 |
| TimesFM, adaptive, unconstrained next-eight shape | −.205 | −.272 | −.293 | −.319 | **−2.989** |
| Bolt, fixed, first-eight shape | .769 | .762 | .774 | .769 | .768 |
| Bolt, adaptive, first-eight shape | .845 | .853 | .858 | .858 | .856 |
| Bolt, adaptive, unconstrained next-eight shape | −.318 | −.270 | −.255 | −.247 | −.415 |

The original 192 daily series showed the same contrast. TimesFM's final versus first site added .117 [.056, .141] to fixed-library early shape. Its weakest first-eight singular sensitivity, in units of one typical state radius, rose from .284 to 4.376. Adapting to each current example offered little first-eight advantage over a frozen library at the final block. This is consistent with a more stable local map to the output there. An independent evaluation of the next eight leads is essential when claiming that a late-layer edit reproduced a mechanism. The fine timing beyond lead eight was wrong even when the early forecast shape closely matched. The five depths correspond to blocks 0/5/10/15/19 in TimesFM and 0/1/3/4/5 in Bolt. [Site capture and intact-chain check](geometry.py) · [Physical interventions](intervene.py)

On hourly traffic, moving the fixed TimesFM receiver from first to final improved short-horizon shape from .234 to .991, conditional on five Bolt donor features. The reverse direction worsened from .638 to .471, with only two frequent TimesFM donor features. On the original VMs, the adaptive TimesFM receiver at its first and final sites scored .903 and .914 against .348 at the middle. This comparison had only four frequent source features, with a wide middle interval [.205, .980]. We therefore froze a separate machine-level confirmation and a full-dose feedback experiment. [New-machine protocol](feedback/PROTOCOL.md)

## Does it help to write to two layers at once?

We optimized at the first and final blocks simultaneously, adding each write to the *live* state so that both interventions could propagate. At a total length budget of 20%, the Bolt recipient scored .860 on additional daily series, compared with .845 first-only and .856 final-only. TimesFM scored .972, against .970 and .972. At a tighter total budget of 5%, the two-site Bolt write scored .683; the first-only site already reached .697. Permitting a quadratic budget allowed the total length across sites to grow and raised the two-site score to .764. Giving two slots to the very same final layer under this budget reached .722. Those raw gains conflate extra available total length and the choice of sites. A sum-of-lengths budget removes that accounting advantage but does not yield a universally superior site pair. This experiment scales a joint quadratic solution to the chosen budget; it does not optimize over all possible allocations of a fixed total length. [Paired-site protocol](PAIRED_SITES.md) · [Actual two-site outcomes and same-site controls](results/paired.json)
