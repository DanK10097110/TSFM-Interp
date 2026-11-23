# Testing a reusable activation direction

Suppose the hypothesis is “this model uses recent momentum.” Raising every forecast tests a directional bias. To test sensitivity to momentum, use rising and falling versions of the same history, apply the same intervention to both, and ask whether their forecast slopes move farther apart.

## Fit once, intervene on new histories

Choose a checkpoint, an internal layer and a token position before examining evaluation outcomes. Supply paired histories with the cue present versus absent, or with opposite signs. Match nuisance variables such as endpoint, scale and noise as closely as the hypothesis permits. The meaning of the direction comes from this experimental contrast.

```python
from tsfm_lens.steering import ActivationContrast, SteeringModel

model = SteeringModel.from_adapter(adapter, quantiles=[0.1, 0.5, 0.9])
profile = ActivationContrast.fit(model, rising_train, falling_train,
                                 layer=layer_name, token=-1, prediction_length=8)
cue_scores = profile.score(new_histories)  # projected activation, centered on the training midpoint
ordinary = profile.predict(new_histories, strength=0)
shifted = profile.predict(new_histories, mode="add", strength=0.5)
amplified = profile.predict(new_histories, mode="amplify", strength=1)
profile.save("momentum.npz")
# In another process, register the same checkpoint, adapter and quantiles.
restored = ActivationContrast.load("momentum.npz", model)
```

The profile records the mean difference **d** between positive and negative hidden states and the midpoint **c** of their two means. Both operations use these same fitted parameters on every history, in one model pass per history.

| Operation | Edit to hidden state **h** | Experimental question |
| --- | --- | --- |
| Add | Add `strength × d`. | Does translating every state in the same direction produce a consistent behavioral bias? |
| Amplify | With unit direction **u** along **d**, add `strength × dot(h − c, u) × u`. | Does increasing the existing signed component change sensitivity to the cue? Strength +1 doubles that component and −1 removes it relative to **c**. |

The edit applies at one input to one layer invoked once per forecast. For a token sequence, `token=-1` means the last model token, which may be a summary token. This Python API requires a deterministic model in evaluation mode and the fitted context length and horizon. The standard run config and studio use the target-fitting controller.

## Read the result as an intervention

Compare zero, positive and negative strengths on held-out pairs. Measure the intended effect in forecast units and check collateral effects such as a shift in average level or interval width. A matched-norm random direction provides a sanity check. For a stronger claim, repeat with multiple controls, independent histories, and contrasts that break correlations between the cue and other properties. A causal effect of changing one hidden state can have several semantic explanations.

The additive construction follows the paired mean-difference idea tested in language models. Forecast behavior requires its own evidence. [Rimsky et al., ACL 2024](https://aclanthology.org/2024.acl-long.828/)

## Two-checkpoint pilot

We chose the middle block and its last token before evaluation. Each of 32 training pairs shared the same sinusoid, noise and final observation. One history had a recent dip followed by a rise, the other a peak followed by a fall. We froze and reloaded the direction, then evaluated another 32 matched pairs generated with a separate seed. The contrast includes both the turning point and the recent slope.

The frozen direction scored the rising history above its falling counterpart in 31/32 evaluation pairs on Bolt and 32/32 on TimesFM. This tests whether the cue can be read from the hidden state. The interventions below test what changing that state does.

Sensitivity here is half the rising-minus-falling difference in forecast tilt, measured relative to the ordinary forecasts. Tilt is the fitted change in median forecast level from the first to the last horizon step. Positive values mean the intervention made forecasts follow the recent slope more strongly.

| Effect at strength +1 | Chronos-Bolt | TimesFM |
| --- | ---: | ---: |
| Sensitivity change after adding the vector | +0.0015 | +0.1285 |
| Average forecast level change after adding the vector | +0.0052 | +0.3332 |
| Sensitivity change after amplifying the component (paired 95% interval) | −0.0068 [−0.0099, −0.0039] | +0.0138 [−0.0152, +0.0426] |

The TimesFM addition changed sensitivity alongside a larger shift in forecast level. Amplification gave a small opposite-sign effect on Bolt and an inconclusive sensitivity effect on TimesFM. Intervals resample evaluation pairs conditional on one fitted vector. A single random direction is a limited control. These synthetic results support studying the intervention and its side effects; a reusable “momentum consideration” would need stronger evidence. [Per-pair evidence, negative doses and random controls](../../../examples/steering/validation/contrast_pilot.json) · [Reproduction script](../../../examples/steering/contrast_pilot.py)
