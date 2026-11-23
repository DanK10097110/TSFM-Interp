# Does one generic historical ramp explain cross-size transfer?

**Much of the early shape recovery survives without a current donor intervention.** We applied a single hidden vector averaged from previous MOMENT-small responses to the same −2°C→+2°C ramp. On 385 untouched station/date inputs, this fixed vector reached .500 centered skill over leads 1–8. The actual history-specific orientation and length reached .669. Giving the generic direction the current donor-derived length raised it to .565. The primary paired advantage of the current orientation at matched length was **.10396 [−.00036, .17851]**. The later eight leads gave clearer evidence for history-specific orientation. [Frozen test](PROTOCOL.md) · [Scores, physical magnitudes and intervals](results/summary.json)

We froze the existing ordinary-hidden-state map from small to large and averaged all 184 historical *small* ramp-induced displacements after transformation. The 46 historical stations and four dates had zero overlap with the 35 new stations and eleven dates. The fixed control uses that stored vector exactly. The matched-length control uses the same direction and the Euclidean norm of the current mapped small displacement. Only the latter needs a current small-checkpoint intervention. Both controls are free of receiving-model ramp responses during their construction. All three candidates are physically written into the same large checkpoint at encoder block 11, position 61, on its ordinary H16 computation. The receiving reference inserts that history’s genuine large-checkpoint ramp-induced state change at this token and keeps the other token states and normalizer ordinary. The endpoint measures a model’s response through that token. [Parent experiment](../new_dates/PROTOCOL.md) · [Physical writes and cap-free vector construction](run.py)

| Target forecast outcome | Current map | Historical mean, fixed length | Historical mean, current length | Paired current minus historical at same length |
|---|---:|---:|---:|---:|
| Leads 1–8, centered shape | .669 [.617, .724] | .500 [.443, .579] | .565 [.497, .653] | .104 [−.00036, .179] |
| Leads 1–8, full | .619 [.541, .681] | .369 [.286, .429] | .425 [.330, .501] | .194 [.104, .279] |
| Leads 9–16, centered shape | .426 [.242, .542] | .225 [.085, .358] | .166 [−.041, .317] | .260 [.144, .397] |
| Leads 9–16, full | .504 [.327, .653] | .294 [.192, .455] | .322 [.184, .455] | .182 [.057, .299] |

Skill is the reduction in total squared error against leaving the receiving forecast unchanged. Centered shape removes each event’s mean over its eight forecast leads. The same 3,000 paired crossed station/date draws and error denominators reproduce the companion experiment's scorer exactly. The raw historical direction achieved 75% of the current map's point-estimated **early squared-error reduction**. Supplying the current length added .065 [.021, .121] to early skill. Current orientation then improved the point estimate on six of eleven individual dates. The pooled matched-length advantage stayed positive when leaving out any one date (range .074–.114) or station (.094–.112), while its joint interval included zero. [Complete per-date results](results/summary.json)

For the later shape the current orientation beat the matched template on ten of eleven dates. The target centered physical RMS was **.1493°C early** and **.0474°C late**. Early RMSE was .0859°C current, .1055°C fixed and .0984°C matched. Late RMSE was .0359°C, .0417°C and .0433°C. Thus the larger late skill contrast occurs on a smaller absolute temperature displacement. The current and length-matched template had similar late forecast RMS (.0311 and .0312°C) despite different late errors. Their early forecast RMS differed (.0889 and .0712°C): equal hidden write length allows different receiving-model gain as well as different output shape. The historical mean’s norm was 8.350 hidden units, compared with individual median norms 11.701 historically and 10.803 on the new panel. The current-to-template hidden cosine had deciles and median .413, .644 and .773. A portable common edit and current-state information both contribute. The evidence for individualized orientation depends on the forecast endpoint. [Waveforms and hidden lengths](results/waves.npz)

At three predetermined new station/date cases, every arm and the receiving reference were rerun through the complete native 24-block model with an identically batched zero-write check. Zero and ordinary forecasts were identical, and the largest complete-model versus cached-prefix discrepancy was **2.87×10⁻⁶°C**. The two synthetic tests passed and four planted regressions failed their intended assertion: leaking new directions into the historical mean, dropping current-length matching, reversing the paired comparison, and omitting within-window centering. [Native replay](results/replay.json) · [Known-answer tests](test_template.py)

To reproduce, first prepare the ordinary and edited state caches with the two [new-station protocols](../../weather/new_dates/PROTOCOL.md) and the [physical-ramp protocol](../new_dates/PROTOCOL.md). From repository root in the existing audited MOMENT environment:

```bash
export PYTHONPATH="/tmp/tsfm_moment_native/vendor:.:tsfm_lens"
export HF_HOME=/tmp/tsfm_moment_native/local_hf HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
study=research.activation_transfer.moment.dictionary_transfer.input_deltas.template
python -m "$study.run"
python -m "$study.score"
python -m pytest -q research/activation_transfer/moment/dictionary_transfer/input_deltas/template/test_template.py
python -m "$study.mutations"
```
