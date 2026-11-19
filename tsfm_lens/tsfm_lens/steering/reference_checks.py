"""Known-answer checks for the meaning of a steering request in forecast units."""
from html import escape

import numpy as np

from .requests import SteeringRequest, context_counterfactual, requested_change


def check_requests():
    """Compare compiled targets and input policies to a four-step worked example."""
    x = np.array([-1., 1., -1., 1.])  # mean 0 and standard deviation 1
    m = np.arange(4.)
    q = np.array([.1, .5, .9])
    base = m[:, None] + [-1., 0., 2.]
    # Direct discrete Gaussian convolution with mirrored endpoints, sigma one.
    grid = np.arange(-4, 5)
    weights = np.exp(-grid**2 / 2)
    filtered = np.convolve(np.pad(m, 4, mode="symmetric"), weights/weights.sum(), mode="valid")
    wave = np.array([0., 1., 0., -1.])
    cases = [
        (SteeringRequest("level", .25), base, [.25]*4),
        (SteeringRequest("trend", .6), base, [-.3, -.1, .1, .3]),
        (SteeringRequest("detrend", 1.), base, [1.5, .5, -.5, -1.5]),
        (SteeringRequest("amplitude", .5), base, [-.75, -.25, .25, .75]),
        (SteeringRequest("seasonality", .5, period=4, phase=0), base, .5*wave),
        (SteeringRequest("deseasonalize", 1., period=4), base+wave[:, None], -wave),
        (SteeringRequest("smooth", 1., smoothing=1.), base, filtered-m),
        (SteeringRequest("roughness", 1., smoothing=1.), base, m-filtered),
        (SteeringRequest("uncertainty", .5), base, [[-.5, 0., 1.]]*4),
        (SteeringRequest("asymmetry", .5), base, [[.5, 0., 1.]]*4),
        (SteeringRequest("quantile", -.25, quantile=.9), base, [[0., 0., -.25]]*4),
        (SteeringRequest("anchor", .5, value=2), base, [1., .5, 0., -.5]),
        (SteeringRequest("mean_revert", 1.), base, [0., -1/3, -4/3, -3.]),
        (SteeringRequest("bounds", lower=.5, upper=2.), base, [.5, 0., 0., -1.]),
        (SteeringRequest("delay", delay=1), base, [0., -1., -1., -1.]),
    ]
    rows = []
    for request, baseline, expected in cases:
        expected = np.asarray(expected)
        if expected.ndim == 1:
            expected = expected[:, None]
        actual = requested_change(baseline, x, q, request)
        rows.append((request.kind, float(np.max(abs(actual-expected)))))
    # A transparent predictor returns its four inputs as its four future medians.
    for request, expected in [
        (SteeringRequest("context_ignore", 1.), [-1., 1., 1., 1.]),
        (SteeringRequest("context_emphasize", .5), [-1., 1., -2., 1.]),
        (SteeringRequest("context_focus", outside_weight=.25), [-.25, .25, -1., 1.]),
        (SteeringRequest("context_recency", np.log(8)), [-.125, .25, -.5, 1.]),
    ]:
        changed = context_counterfactual(x, request)
        ordinary = x[:, None] + [-1., 0., 2.]
        actual = requested_change(ordinary, x, q, request,
                                  counterfactual_forecast=changed[:, None]+[-1., 0., 2.])
        error = np.max(abs(actual - (expected-x)[:, None]))
        rows.append((request.kind, float(error)))
    return rows


def report_controls():
    rows = check_requests()
    passed = sum(error < 1e-9 for _, error in rows)
    table = "".join(f"<tr><td>{escape(kind)}</td><td>{error:.3g}</td></tr>" for kind,error in rows)
    return (f"<details><summary>Known-answer target checks · {passed}/{len(rows)} passed</summary>"
            "<p>Four-step examples test the target definitions independently of model fitting. "
            "The context alternates −1 and +1. Median forecasts increase from 0 to 3, with a planted "
            "four-step wave for the season removal check. For context policies, a transparent predictor "
            "copies its inputs. The table gives the largest absolute difference from the analytic "
            "answer across steps and channels. The pass threshold is 1e−9.</p>"
            "<table><tr><th>Direction</th><th>Maximum absolute error</th></tr>" + table + "</table></details>"
            "<details><summary>Use the fitted model on another input</summary>"
            "<p>Save a model/request ZIP using its export link. With this toolkit and the base checkpoint "
            "available, call the predictor on a new context of the same length.</p>"
            "<pre>from tsfm_lens.steering import SteeredForecaster\n"
            "model = SteeredForecaster.load('profile.zip')\n"
            "comparison = model.compare(my_context)\n"
            "ordinary, steered = comparison.baseline, comparison.forecast</pre>"
            "<p>To change directions and doses, launch the local desktop studio with your run YAML. "
            "Choose a reference input or paste your own. The export dialog can include cached weights.</p>"
            "<pre>python -m tsfm_lens.steering.studio --config CONFIG.yaml</pre></details>")
