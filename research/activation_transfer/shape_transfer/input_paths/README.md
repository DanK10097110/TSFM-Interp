# An absolute level shift can bypass the middle token states

Adding 0.05 history standard deviations to **every observation** lifted all eight forecast leads by 0.05 on all 128 daily and 128 weekly series in each of three checkpoints. The largest absolute discrepancy across those 6,144 forecast values was 0.000001956. Restoring the entire middle-block token tensor to its value on the clean history preserved the shift in every checkpoint. Median relative movement of the original token tensors was below 0.00000041 in each of the six model/domain cells. Thus shared behavior under an absolute change of origin supplies almost no evidence about a shared hidden direction at this site. The earlier hidden lift used an intervention on an unchanged history, a separate causal pathway.

Changing only recent observations distinguishes that case from a *relative* level or shape change. The table gives the signed projection of the isolated patch effect on the model's ordinary input-edit effect, pooled across the 128 daily histories and eight leads. A value of one recovers the amplitude along the actual response. For each patch we took the state from the edited run and placed it in the clean run. Inputs use all 192 observations. The step and zero-mean ramp change just the final 32.

| Checkpoint | Recent step, last token | Recent step, all tokens | Recent ramp, last token | Recent ramp, all tokens |
|---|---:|---:|---:|---:|
| Chronos-Bolt-small | .014 | .611 | .026 | .968 |
| TimesFM 2.5 200M | .597 | .597 | .968 | .968 |
| Timer-base 84M | .581 | .580 | .973 | .973 |

The weekly ramp projections for the last token were .025, .955 and .959 respectively. Bolt appends a regression token beyond its final observed patch. A natural recent change can run through the middle of this model while barely moving the forecast through the last-token pathway used in the original steering test. This motivated a direct test at the data-bearing tokens. The remaining and transplanted gains need not sum to one in a nonlinear model. Each recorded output contains all eight leads in the model's ordinary forecast units. [Data-token steering experiment](../token_position/README.md) · [Both swap directions, relative state movement and all domains](results/results.json) · [Protocol](PROTOCOL.md) · [Production intervention](run.py)
