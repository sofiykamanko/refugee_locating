'''Run before every pull request: python -m pytest (well under a minute).'''
import time
from pathlib import Path

import numpy as np

from refugee_sim.generator import capacities
from refugee_sim.policy import assign_exact, assign_exact_scipy
from refugee_sim.simulation import PLACEMENT_COLUMNS, expand, load_config, run_simulation

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
