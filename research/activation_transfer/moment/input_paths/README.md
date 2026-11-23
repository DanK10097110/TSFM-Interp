# Why a shared response to adding two degrees can be misleading

On 184 real 2026 station histories, adding 2°C to all 480 input observations shifted the 16-step forecasts of both MOMENT-small and large by 2°C. Swapping only the final reverse-normalization statistics reproduced essentially the entire shift. A different experiment, adding 2°C only to the **latest eight days**, produced visibly different forecasts and internal pathways.

| Released checkpoint and synthetic change to the past | Mean first-eight change | Mean second-eight change | Share through middle observed patches | Share through future patches | Share through reverse normalization |
|---|---:|---:|---:|---:|---:|
| Small, last eight days +2°C | −0.019°C | +0.063°C | .04 | .98 | −.01 |
| Large, last eight days +2°C | +0.864°C | +0.614°C | .63 | .32 | .05 |
| Small, last eight ramp from −2 to +2°C | +0.008°C | +0.019°C | .03 | .97 | .00 |
| Large, same ramp | +0.362°C | +0.267°C | .64 | .36 | .00 |

The decomposition measures the *forecast's response* to changing a historical input. It does not predict what temperature would have occurred in a counterfactual climate. Every cell uses the same history, horizon and native scaling. For each input change, we transferred the middle-block observed states, the two future-mask states and the denormalization factors independently between clean and modified runs. We evaluated all eight combinations. Shares are signed projections of Shapley contributions onto the actual first-eight response; they can cancel and need not equal a percentage of variance. Small's near-zero signed mean can coexist with measurable per-station waveform changes. The data include 46 spatially separated stations with four fixed dates each. [Protocol](PROTOCOL.md) · [Raw effects, magnitudes, station-blocked intervals and cross-checkpoint comparisons](results/summary.json)

A simple shared level response therefore has two possible readings. A change of origin in the entire series travels through normalization. A recent local warming instead changes the encoded series; in large, most of its near-term response at this layer still traveled through the observed tokens. At the middle of small, most of the measured response was already in the future mask states. These statements concern a specific token partition and a specific released readout. [Native patch and normalization code](https://github.com/moment-timeseries-foundation-model/moment/blob/main/momentfm/models/moment.py) · [Earlier three-model level-path control](../../shape_transfer/input_paths/README.md)
