'''Run before every pull request: python -m pytest.'''
import time
from pathlib import Path

import numpy as np

from refugee_sim.generator import ORIGIN_IDX, capacities, initial_stock, sample_attributes
from refugee_sim.policy import (PolicyModel, assign_exact, assign_exact_scipy, assign_forward, encode,
                                share_increase)
from refugee_sim.simulation import PLACEMENT_COLUMNS, _merge, expand, load_config, run_simulation

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
METRICS = ['welfare_algo', 'welfare_random', 'welfare_oracle', 'gain_pct', 'oracle_gain_pct',
           'captured_pct', 'promised_gain_pct', 'regret', 'calib_gap', 'rmse_all', 'auc',
           'origin_gap', 'sex_gap', 'mean_z_placed', 'realised']


def test_smoke_run():
    config = load_config(CONFIGS / 'smoke.yaml')
    start = time.time()
    df = run_simulation(config, seed=0)
    assert time.time() - start < 60
    assert list(df['round']) == [1, 2]
    assert df['run_key'].nunique() == 1
    assert df[METRICS].drop(columns='captured_pct').notna().all().all()
    assert (df[PLACEMENT_COLUMNS].sum(axis=1) == config['cohort_size']).all()
    assert df.equals(run_simulation(config, seed=0))   


def test_fast_solver_matches_scipy():
    rng = np.random.default_rng(0)
    scores, cap = rng.random((500, 16)), capacities(500)
    a, b = assign_exact(scores, cap), assign_exact_scipy(scores, cap)
    assert np.isclose(scores[np.arange(500), a].sum(), scores[np.arange(500), b].sum())
    assert (np.bincount(a, minlength=16) == cap).all()


def test_experiment_configs():
    assert len(expand(load_config(CONFIGS / 'main.yaml'))) == 45
    assert len(expand(load_config(CONFIGS / 'history.yaml'))) == 6
    assert len(expand(load_config(CONFIGS / 'forms.yaml'))) == 10
    assert len(expand(load_config(CONFIGS / 'community_model.yaml'))) == 54
    assert len(expand(load_config(CONFIGS / 'forward_looking.yaml'))) == 18
    assert len(expand(load_config(CONFIGS / 'learning_delta.yaml'))) == 20
    assert len(expand(load_config(CONFIGS / 'model_class.yaml'))) == 18


def test_community_features_run():
    config = _merge(load_config(CONFIGS / 'smoke.yaml'),
                    {'policy': {'features': 'community_recent', 'retrain': 'every_round'}})
    df = run_simulation(config, seed=0)
    assert (df['features'] == 'community_recent').all()
    assert df['gain_pct'].notna().all()


def test_community_matrix_matches_single_predictions():
    '''predict_matrix must give person i in state k the community value C[k, origin_i].'''
    rng = np.random.default_rng(0)
    people = sample_attributes(800, rng)
    N, T = initial_stock()
    C, R = N / T[:, None], rng.random(N.shape) / 100
    origins = people['country_of_origin'].map(ORIGIN_IDX).to_numpy()
    states = rng.integers(0, 16, len(people))
    y = rng.random(len(people)) < 0.2 + 5 * C[states, origins]
    model = PolicyModel(0, 'community_recent').fit(
        people, states, y, {'C': C[states, origins], 'R': R[states, origins]})
    matrix = model.predict_matrix(people, {'C': C, 'R': R})
    for k in (0, 7, 15):
        rows = np.column_stack([encode(people), np.full(len(people), k), C[k, origins], R[k, origins]])
        assert np.allclose(matrix[:, k], model.clf.predict_proba(rows)[:, 1])


def test_forward_placement():
    '''lookahead 0 is the greedy placement; with lookahead the placement scores higher on its own
    objective (this cohort + effect on future cohorts) and still meets the quotas.'''
    rng = np.random.default_rng(0)
    n, K = 600, 16
    scores, cap = rng.random((n, K)) * 0.2, capacities(n)
    origins = rng.integers(0, 9, n)
    slope = -rng.random((K, 9))
    N, T = initial_stock()
    dC = share_increase(N, T, 50) * 50
    assert (assign_forward(scores, cap, origins, slope, dC, 0) == assign_exact(scores, cap)).all()

    def objective(pi, lookahead):
        counts = np.zeros((K, 9))
        np.add.at(counts, (pi, origins), 1)
        return scores[np.arange(n), pi].sum() + lookahead * (counts * slope * dC * counts).sum()

    forward = assign_forward(scores, cap, origins, slope, dC, 10)
    assert (np.bincount(forward, minlength=K) == cap).all()
    assert objective(forward, 10) > objective(assign_exact(scores, cap), 10)


def test_forward_run():
    config = _merge(load_config(CONFIGS / 'smoke.yaml'),
                    {'policy': {'features': 'community', 'planner': 'model', 'lookahead': 3}})
    df = run_simulation(config, seed=0)
    assert (df['lookahead'] == 3).all() and df['gain_pct'].notna().all()
    assert df['delta_hat'].notna().all()


def test_model_classes_run():
    for name in ('hybrid', 'lasso'):
        config = _merge(load_config(CONFIGS / 'smoke.yaml'), {'policy': {'model': name, 'lookahead': 3}})
        df = run_simulation(config, seed=0)
        assert (df['model'] == name).all() and df['gain_pct'].notna().all()
