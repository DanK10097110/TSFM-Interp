"""Regression checks for conserved input moments, signed outcomes and paired controls."""
from pathlib import Path

import numpy as np

from research.activation_transfer.controls.analyze import compare, holm, stratified_draws
from research.activation_transfer.controls.collect import edit_histories


def test_controls_preserve_stated_coordinates():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(5,128))
    x[0,-32:] = 1.
    pair, rms, summaries = edit_histories(x)
    np.testing.assert_allclose(pair[:,5].mean(-1), np.repeat(x.mean(-1)[:,None], 2, axis=1), atol=1e-7)
    np.testing.assert_allclose(pair[:,5].std(-1), np.repeat(x.std(-1)[:,None], 2, axis=1), atol=1e-7)
    np.testing.assert_allclose(pair[:,5,:,-1], np.repeat(x[:, -1,None], 2, axis=1), atol=1e-7)
    np.testing.assert_allclose(pair[:,4,1]-pair[:,4,0], .7, atol=2e-7)
    assert rms[0,5] == 0 and (rms[1:,5] > 0).all()
    assert pair.shape == (5,6,2,128) and summaries.shape == (5,10)


def test_conditional_source_recovers_planted_signed_response():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(160,10))
    source = rng.normal(size=(160,4,16))
    destination = 3 * source
    fit = np.arange(60)
    test = np.arange(60,160)
    row = compare(x[fit], x[test], source[fit], destination[fit], source[test], destination[test],
                  component="shape", use_stats=True, seed=19, draws=200, permutations=99)
    np.testing.assert_allclose(row["gain"], 1.)
    np.testing.assert_allclose(row["calibration_scalar"], 3.)
    assert row["perm_p"] == .01 and row["null_mean"] < 0


def test_donor_mapping_respects_roughness_and_holm():
    x = np.random.default_rng(73).normal(size=(123,10))
    group = np.digitize(x[:,8], np.quantile(x[:,8], [.25,.5,.75]))
    for perm in stratified_draws(x, np.random.default_rng(18), 3):
        np.testing.assert_array_equal(group, group[perm])
        assert len(set(perm)) == len(group)
    np.testing.assert_allclose(holm([.98,.03,.002]), [.98,.06,.006])


def test_local_forecasters_translate_exactly():
    from research.activation_transfer.controls.classical import classical_response
    rng = np.random.default_rng(76)
    x = rng.normal(size=(3,128))
    pairs = np.stack([x-.35, x+.35], axis=1)[:,None]
    ys = classical_response(pairs)
    for response in ys.values():
        np.testing.assert_allclose(response, .35, atol=2e-6)


def test_nonnegative_two_source_response():
    from research.activation_transfer.controls.classical import fit_predict
    rng = np.random.default_rng(731)
    sources = [rng.normal(size=(21, 4, 16)) for _ in range(2)]
    target = .8*sources[0] + .3*sources[1]
    predictions, coefs = fit_predict(sources, sources, target)
    np.testing.assert_allclose(predictions, target)
    np.testing.assert_allclose(coefs, np.tile([.8,.3],(4,1)))
    # Ordinary least squares fits a negative coefficient in this orthogonal design.
    axes = np.broadcast_to(np.eye(2), (8, 1, 2, 2))
    orthogonal = [axes[...,0], axes[...,1]]
    _, constrained = fit_predict(orthogonal, orthogonal, .8*orthogonal[0]-.3*orthogonal[1])
    np.testing.assert_allclose(constrained, [[.8,0.]])


def test_two_way_bootstrap_and_matched_donors():
    from research.activation_transfer.controls.seattle import bootstrap_weights, crossed_gain, donor_test
    rng = np.random.default_rng(321)
    a = rng.normal(size=(18, 9, 4, 16))
    err = (a**2).mean(-1)
    summary = crossed_gain(err, .75*err, bootstrap_weights(18, 9))
    np.testing.assert_allclose(summary["gain"], .25)
    np.testing.assert_allclose(summary["ci95"], [.25,.25])
    checks = donor_test(a, a, np.zeros_like(err))
    assert checks["date"]["p"] == .001 and checks["sensor"]["p"] == .001
    # A synchronized error on one date is invisible to resampling sensors alone.
    base = np.ones((8,4,3))
    dated_errors = np.zeros_like(base)
    dated_errors[:,3] = 4.
    dated = crossed_gain(base, dated_errors, bootstrap_weights(8,4))
    np.testing.assert_allclose(dated["gain"], 0.)
    np.testing.assert_allclose(dated["ci95"], [-2.,1.])


def test_shape_only_addition_respects_level_orthogonality():
    from research.activation_transfer.controls.shape import evaluate
    rng = np.random.default_rng(615)
    # Mean and centered waveform carry nine and one units of energy, respectively.
    level = rng.choice([-3.,3.], size=(32,4,1))
    shape = rng.choice([-1.,1.], size=(32,4,1))*np.tile([1.,-1.],8)
    true = level+shape
    predicted = dict(shape_classical=np.zeros_like(shape), level_classical=np.zeros_like(level),
                     shape_neural=shape, level_neural=level, full_neural=true, full_classical=np.zeros_like(true))
    scores = evaluate(true, predicted, seed=325)["schemes"]
    np.testing.assert_allclose(scores["both_components"]["gain"], 1.)
    np.testing.assert_allclose(scores["shape_only"]["gain"], .1)
    np.testing.assert_allclose(scores["level_only"]["gain"], .9)


def test_primary_manifests_are_distinct_and_response_labels_agree():
    import json
    here = Path(__file__).resolve().parent
    items = json.loads((here/"items.json").read_text())["rows"]
    assert len(items) == len({r["item"] for r in items}) == len({r["sha256"] for r in items}) == 576
    with np.load(here/"responses.npz", allow_pickle=False) as arrays:
        assert arrays["item"].tolist() == [r["item"] for r in items]
        assert arrays["split"].tolist() == [r["split"] for r in items]


def test_atom_audit_distinguishes_row_space_from_shrinkage():
    from research.activation_transfer.controls.atom_audit import spectral_support
    a = 1e-2
    states = np.array([[1.,0,0],[-1,0,0],[0,a,0],[0,-a,0]])
    dirs = np.eye(3)
    shrinker = np.diag([1/1.01, a*a/(a*a+.01), 0])
    audit = spectral_support(states, dirs, shrinker)
    np.testing.assert_allclose(audit['row_span_norm_fraction'], [1,1,0])
    np.testing.assert_allclose(audit['ridge_norm_fraction'], np.diag(shrinker))
    assert audit['modes_above_half_response'] == 1 and audit['rank'] == 2


def test_atom_audit_repeated_rows_and_inactive_atoms():
    from research.activation_transfer.controls.atom_audit import numerical_summary
    row = np.array([0,0,1,2]); atom = np.array([0,1,1,0])
    effect = np.zeros((3,3,2)); effect[row, atom] = 2.
    noise = np.array([[0.,0.],[.5,0.],[0,0],[0,0]])
    report = numerical_summary(noise,effect,row,atom,np.array(['a','a','b']),['a','b'],3)
    assert report['a']['n_pairs'] == 3 and report['b']['n_pairs'] == 1
    assert report['a']['pair_effect_rms_within_10x_noise'] == 1
    assert report['a']['per_atom'][1]['n'] == 2 and report['a']['per_atom'][2] is None
    assert report['a']['per_atom'][1]['noise_rms'] == .25


def test_dictionary_scramble_preserves_empirical_moments_and_changes_atoms():
    from research.activation_transfer.controls.dictionary_controls import make_bases
    rng = np.random.default_rng(370)
    d = rng.normal(size=(12,18)) + .35
    x = rng.normal(size=(180,18)) @ np.diag(np.linspace(.01,1.,18))
    bases, metadata = make_bases(x, d, 144)
    moved = bases['decoder_covariance']
    np.testing.assert_allclose(moved.mean(0), d.mean(0), atol=1e-12)
    np.testing.assert_allclose(np.cov(moved.T), np.cov(d.T), atol=1e-12)
    assert np.linalg.norm(moved-d) > 1 and metadata['n_eligible'] == 12
    np.testing.assert_allclose(bases['pca']@bases['pca'].T, np.eye(12), atol=1e-12)
    for a in bases.values():
        assert a.shape == d.shape


def test_dictionary_hidden_recovery_penalizes_overdosing_a_correct_axis():
    from research.activation_transfer.controls.dictionary_controls import vector_metrics
    d = np.eye(4)
    out = vector_metrics(d,2*d)
    np.testing.assert_allclose(out['cosine'],1.)
    np.testing.assert_allclose(out['recovery'],0.)
