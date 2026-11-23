# Copying the donor's tilt and making the requested tilt are different tasks

We kept the original middle sites and original fixed source directions. For each held-out history, an adaptive translator used both models' ordinary-state derivatives to predict what the donor direction would do, found a recipient vector with a similar predicted eight-lead effect, and applied it to the actual recipient. We varied whether the target write was restricted to its top 24 clean PCs and whether the translator used paired calibration derivatives or derivatives on the current history. All methods here use the same relative damping and physical norm cap. [Protocol](PROTOCOL.md) · [Live forecast code](run.py)

| Daily outcome | TimesFM → Bolt, tilt | TimesFM → Bolt, sine | Bolt → TimesFM, tilt | Bolt → TimesFM, sine |
|---|---:|---:|---:|---:|
| Copy the actual source forecast change, all eight leads | .990 | .993 | .971 | .975 |
| Copy its shape after separately removing each horizon mean | .988 | .991 | .986 | .985 |
| Give only odd lead derivatives; copy the *centered* even leads | .793 | .805 | .959 | .943 |
| Meet the user's requested shape with the copied change | .111 | .059 | .289 | .220 |
| Meet the same request by solving for it directly in the local recipient | .783 | .717 | .984 | .878 |
| Meet the request with the original calibrated clean-state map | .000 | .000 | .044 | .024 |

On weekly series, recovery of the original donor change was .977–.993 for the four pair/request combinations. Local recipient control of the external goal scored .798–.988. Restricting the history-specific write to the 24-PC recipient subspace still gave .878–.933 full *donor-effect recovery* on the four daily cases. Adapting to the current history helped even in that restricted space. The generator of the fixed donor direction already produced a relatively small tilt or sine. A translator can faithfully copy its weakness. A direct recipient solve knows the requested waveform and spends its own available control authority on that goal. Its large improvement answers whether the receiver *can* produce the requested shape. A fixed portable donor direction would have a lower access cost than either history-specific procedure.

Zero is the skill for leaving the forecast alone. Donor reproduction uses the measured donor effect as denominator; request recovery uses the RMS-0.05 objective. The results compare actual forecast changes, independently of future forecast accuracy. On the original site, weekly and daily outcomes, limits, fitted-site replay and calibrated comparator are available per cue. [Scores, partial information and paired confidence intervals](results/results.json)
