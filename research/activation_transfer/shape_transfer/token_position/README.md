# Aligning the observed token improves some comparisons, with a clear limit

The original Chronos-Bolt middle-site intervention used its trailing regression token. TimesFM's last token was the patch that contains the latest observations. We refitted the same native steering and clean-state translation using Bolt's final observed 16-step patch, then its last two observed patches together. Two patches cover the same 32 observations as one TimesFM patch. All interventions use the same model blocks, cohorts, 24-PC native training, recipient-unit normalization, scalar calibration and four shuffled-correspondence controls. [Protocol](PROTOCOL.md) · [Live patches and token registration](run.py)

| Daily requested change | Original Bolt regression token | Last Bolt data token | Last two Bolt data tokens |
|---|---:|---:|---:|
| TimesFM → Bolt, lift | .822 | .852 | .823 |
| TimesFM → Bolt, tilt | .000 | .000 | .000 |
| TimesFM → Bolt, sine | .000 | .000 | .000 |
| Bolt → TimesFM, lift | .987 | .973 | .988 |
| Bolt → TimesFM, tilt | .044 | .068 | −.070 |
| Bolt → TimesFM, sine | .024 | .047 | .008 |

The receiving Bolt could produce positive native shape changes on its data tokens. On its final token, daily native skill was .304 for tilt and .173 for sine. The clean-state translation from TimesFM still gave roughly zero for both. In the reverse direction the final Bolt data token helped modestly on both domains. Weekly tilt and sine increased from .049 and .011 to .068 and .054. Extending the patch to the full corresponding 32 steps reduced those scores to −.069 and −.002. Thus the physical mismatch matters for how natural changes propagate and for some learned controls. Simply matching observation spans did not rescue a generic shape translator. [Paired intervals, doses and native controls](results/results.json) · [Natural input swap at the same blocks](../input_paths/README.md)
