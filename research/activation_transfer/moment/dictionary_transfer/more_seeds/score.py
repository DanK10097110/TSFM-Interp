"""Score the three receiver initializations on exactly the same source interventions."""
import json

import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts, terms
from research.activation_transfer.moment.dictionary_transfer.more_seeds.fit import SEEDS
from research.activation_transfer.moment.dictionary_transfer.more_seeds.run import ARMS, ORIGINAL, OUT, SOURCE, original_data
from research.activation_transfer.moment.dictionary_transfer.weather.score import centered, cube, infer, interval, matched_effect
from research.activation_transfer.moment.dictionary_transfer.weather.study import OUT as WEATHER_OUT
from research.activation_transfer.shape_transfer.support_confirmation.noaa.score import eligible_mask


def cell(point, stations, crossed):
    return dict(point=float(point), station95=interval(stations), crossed_station_feature95=interval(crossed))


def summarize_receiver(seed, orig, chosen, weights, feature_weights):
    r, c = orig['rows'], orig['cols']
    station = orig['station']
    active = orig['z'] > 0
    donor = matched_effect(orig, 'donor')
    keep = np.isin(c, chosen)
    if seed == ORIGINAL:
        waves = {arm: matched_effect(orig, arm) for arm in ARMS}
        match = orig
    else:
        waves = {}
        for arm in ARMS:
            with np.load(OUT / f'{seed}_{arm}.npz') as f:
                np.testing.assert_array_equal(f['rows'], r)
                np.testing.assert_array_equal(f['cols'], c)
                waves[arm] = f['effect']
        with np.load(OUT / f'match_{seed}.npz') as f:
            match = dict(f)
    refs = {}
    for measure, norm, center in [('history_sd_centered', True, True), ('celsius_centered', False, True), ('celsius_full', False, False)]:
        def transform(x):
            x = x / orig['input_sd'][r, None] if norm else x
            return centered(x) if center else x
        ref = transform(donor)
        goal = cube(ref, r, c, active.shape)[:, chosen]
        arms, fits = {}, {}
        for name, wave in waves.items():
            pred = transform(wave)
            estimate = cube(pred, r, c, active.shape)[:, chosen]
            mat = terms(active[:, chosen], estimate, goal)
            moment = np.stack([mat[station == i].sum(0) for i in np.unique(station)])
            point, sb, fb, per = infer(moment, weights, feature_weights)
            fits[name] = (point, sb, fb)
            truth, estimate, cross, n = moment.sum((0, 1))
            residual = truth + estimate - 2*cross
            arms[name] = dict(macro_skill=cell(point, sb, fb), pooled_skill=float(1-residual/truth),
                              donor_rms=float(np.sqrt(truth/n)), recipient_rms=float(np.sqrt(estimate/n)),
                              error_rms=float(np.sqrt(max(0, residual)/n)), positive_count=int((per>0).sum()),
                              median_donor_skill=float(np.median(per)),
                              per_donor=dict(zip(map(str, orig['atoms'][chosen]), map(float, per))))
        contrasts = {}
        for first, second in [('group16', 'corr_own'), ('group16', 'corr_forced'), ('corr_forced', 'corr_own')]:
            diff = [a-b for a, b in zip(fits[first], fits[second])]
            contrasts[f'{first}_minus_{second}'] = cell(*diff)
        refs[measure] = dict(arms=arms, contrasts=contrasts)
    co = match['target_z'][r,c] > 0
    # Verify the archived control's eligible-event support and raw physical gates.
    np.testing.assert_allclose(waves['corr_own'][~co], 0, atol=1e-5)
    return dict(eligible_events=int(keep.sum()), cofires=int(np.sum(keep & co)),
                median_training_correlation=float(np.median(match['corr'][chosen])),
                median_group_hidden_r2=float(np.median(match['group_r2'][chosen])), metrics=refs)


def score():
    orig = original_data()
    station = orig['station']
    chosen = np.flatnonzero(eligible_mask(orig['z'] > 0, station))
    assert len(chosen) == 12 and len(station) == 184 and len(np.unique(station)) == 46
    w = resampling_counts(46, draws=3000)
    fw = np.random.default_rng(26112283).multinomial(len(chosen), np.full(len(chosen), 1/len(chosen)), 3000)
    report = dict(source_seed=SOURCE, nominated_atoms=orig['atoms'].tolist(), eligible_atoms=orig['atoms'][chosen].tolist(),
                  dates=4, stations=46, source_firings=len(orig['rows']), eligible_firings=int(np.isin(orig['cols'], chosen).sum()),
                  source_weather_checkpoint_sha256=str(orig['checkpoint_sha256']), source_context_sha256=str(orig['context_sha256']),
                  source_sae_sha256=str(orig['source_sae_sha256']), receivers={})
    previous = json.loads((WEATHER_OUT / 'summary.json').read_text())[str(SOURCE)]['statistics']
    for seed in (ORIGINAL, *SEEDS):
        entry = summarize_receiver(seed, orig, chosen, w, fw)
        if seed == ORIGINAL:
            for name, sub in entry['metrics'].items():
                for arm in ARMS:
                    prior = previous[name]['arms'][arm]
                    now = sub['arms'][arm]
                    np.testing.assert_allclose([now['macro_skill']['point'], *now['macro_skill']['station95'],
                                                *now['macro_skill']['crossed_station_feature95'], now['recipient_rms']],
                                               [prior['skill'], *prior['station95'], *prior['crossed95'], prior['candidate_pooled_rms']],
                                               atol=1e-12)
        report['receivers'][str(seed)] = entry
    # Descriptive stability across the three fixed fits; no receiver-seed resampling.
    comparison = {}
    for single in ('corr_own', 'corr_forced'):
        wins = []
        for entry in report['receivers'].values():
            arms = entry['metrics']['history_sd_centered']['arms']
            wins.append([arms['group16']['per_donor'][str(a)] > arms[single]['per_donor'][str(a)]
                         for a in report['eligible_atoms']])
        comparison['group_beats_'+single] = dict(total_cells=int(np.sum(wins)),
            donors_in_all_three=int(np.all(wins, axis=0).sum()), donors_by_seed=np.sum(wins, axis=1).tolist())
    report['paired_donor_seed_counts'] = comparison
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    score()
