# How forecast steering works under the hood

The run config and studio use `DynamicSteerer`. It learns **where a small edit can influence a forecast**, then chooses **how much to edit for each new history**. The model weights stay fixed. An internal edit temporarily adds a vector to the activation entering a layer during a forecast pass.

### Fitting a profile

1. **Turn the request into a target.** Suppose the ordinary forecast is `[100, 102, 101]` and you want to raise it by two. The desired change is `[2, 2, 2]`. The controller specifies a change for every selected step and quantile, including zero where a value should stay put. This makes unwanted movement count toward the error. A mixed request adds its component changes relative to the same ordinary forecast.
2. **Measure which edits move which outputs.** On calibration histories, the controller tries small changes at candidate layers or at the input. Paired positive and negative nudges estimate how each direction affects the forecast. When the final forecast layer is identifiable, its weights suggest promising directions directly. On an unfamiliar linear layer, three probe directions can identify the output rows and a fourth checks the match. Limiting the number of directions (the *rank*) keeps the search affordable.
3. **Choose a location on separate histories.** The controller tries each candidate’s proposed edit through the full model and compares the achieved change with the target. It selects by validation error, preferring a smaller edit when candidates are essentially exact. The saved profile contains the location, directions and measured responses. Separate validation histories test whether the choice works beyond the examples used to construct it.

### Steering a new history

The controller computes the ordinary forecast and uses the profile to solve a bounded least-squares problem to find the combination of directions that comes closest to this history’s requested change. The *radius* caps the total edit size. Because the measured response is a local approximation, the controller runs the model with the proposed edit, tries smaller steps if needed, and accepts only an improvement. It can remeasure and refine within the same radius.

### Checking the result

The report reapplies the saved edit on a held-out history. Verification requires a matching replay and a nonzero requested change reached within tolerance. In the example above, raising every step by one gives 50% relative error, even with perfect replay. Observed futures enter the separate forecast-accuracy scores.

Full-horizon input policies such as `context_ignore` transform the history and forecast from it directly. Their replay checks that the input transformation reproduced the result.

[Request definitions](requests.py) · [Controller and replay](dynamic.py) · [Bounded solver](controller.py) · [Report checks](report.py)
