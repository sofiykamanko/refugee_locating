'''Feedback-loop simulation, yaml configs and csv results.

simulate() is the v2 loop from refugee_experiments_v2.ipynb, section 4, unchanged.
run_simulation(config, seed) is the single entry point; run_experiment() runs a config file.

    python -m refugee_sim.simulation configs/main.yaml
'''
import copy
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from .generator import (INTERACTION_SCENARIOS, ORIGINS, ORIGIN_IDX, STATE_QUOTA, STATES, GroundTruth,
                        assign_states, bernoulli, capacities, initial_stock, normalise,
                        sample_attributes, update_community)
from .estimation import estimate_delta
from .policy import PolicyModel, assign_exact, assign_forward, community_slope, share_increase


PLACEMENT_COLUMNS = [f"place__{s}__{o}" for s in STATES for o in ORIGINS]


def simulate(truth, rounds=10, cohort_size=2000, n_hist=10000, retrain_every=None,
                   window=None, scale=50, seed=0, log_placements=True, features="base",
                   planner="model", lookahead=0):
    """One run of the feedback loop (11.11 version, connected to the corrected ground truth).

    retrain_every  None = fixed model, 1 = every round, 3 = every third round
    scale          real people represented by one simulated person
    features       what the policy model sees about the community (policy.FEATURE_SETS)
    planner        who places people: "model" (learned), "oracle" (knows the truth) or "random" (quota)
    lookahead      future rounds the placement accounts for (0 = greedy, policy.assign_forward)
    The people and the noise depend only on `seed`, so all settings see the same cohorts.
    """
    rng_hist = np.random.default_rng(seed)
    rng_stream = np.random.default_rng(seed + 10_000)
    q = normalise(STATE_QUOTA)
    cap = capacities(cohort_size)
    idx = np.arange(cohort_size)

    N, T = initial_stock()
    C, R = N / T[:, None], np.zeros_like(N)

    # Round 0: historical data under random exact quota assignment at C0
    people0 = sample_attributes(n_hist, rng_hist)
    u0 = rng_hist.normal(0, truth.sigma_u, n_hist)
    s0 = assign_states(n_hist, rng_hist, mode="exact")
    y0 = bernoulli(rng_hist, truth.probability(people0, s0, u0, C, R), n_hist).astype(int)
    history = [(people0, s0, y0, placed_community(people0, s0, C, R))]
    model = PolicyModel(seed, features).fit(*history[0])

    logs, placements = [], []
    for r in range(1, rounds + 1):
        if retrain_every and r > 1 and (r - 1) % retrain_every == 0:
            used = history if window is None else history[-window:]
            model = PolicyModel(seed + r, features).fit(
                pd.concat([h[0] for h in used], ignore_index=True),
                np.concatenate([h[1] for h in used]),
                np.concatenate([h[2] for h in used]),
                {f: np.concatenate([h[3][f] for h in used]) for f in ("C", "R")})

        people = sample_attributes(cohort_size, rng_stream)
        u = rng_stream.normal(0, truth.sigma_u, cohort_size)
        P = truth.probability_all_states(people, u, C, R)        # truth under the CURRENT world

        scores = model.predict_matrix(people, {"C": C, "R": R})
        origin_idx = people["country_of_origin"].map(ORIGIN_IDX).to_numpy()
        pi = place(people, u, scores, P, C, R, N, T, origin_idx, cap, model, truth,
                   planner, lookahead, scale, rng_stream)                     # algorithm
        pi_star = assign_exact(P, cap)                            # greedy oracle (benchmark)
        y = (rng_stream.random(cohort_size) < P[idx, pi]).astype(int)

        p_algo, p_star, p_rand = P[idx, pi], P[idx, pi_star], P @ q
        origin = people["country_of_origin"].to_numpy()
        male = (people["sex"] == "male").to_numpy()
        by_origin = pd.Series(p_algo).groupby(origin).agg(["mean", "size"])
        by_origin = by_origin[by_origin["size"] >= 50]
        pred_chosen = scores[idx, pi]
        gain = 100 * (p_algo.mean() / p_rand.mean() - 1)
        oracle_gain = 100 * (p_star.mean() / p_rand.mean() - 1)
        logs.append({
            "round": r,
            "welfare_algo": p_algo.mean(),
            "welfare_random": p_rand.mean(),
            "welfare_oracle": p_star.mean(),
            "gain_pct": gain,
            "oracle_gain_pct": oracle_gain,
            "captured_pct": 100 * gain / oracle_gain if oracle_gain > 0 else np.nan,
            "promised_gain_pct": 100 * (pred_chosen.mean() / p_rand.mean() - 1),
            "regret": p_star.mean() - p_algo.mean(),
            "calib_gap": pred_chosen.mean() - p_algo.mean(),
            "rmse_all": np.sqrt(np.mean((scores - P) ** 2)),
            "auc": roc_auc_score(y, pred_chosen),
            "origin_gap": by_origin["mean"].max() - by_origin["mean"].min(),
            "sex_gap": p_algo[male].mean() - p_algo[~male].mean(),
            "mean_z_placed": ((C[pi, origin_idx] - truth.c_mean) / truth.c_sd).mean(),
            "realised": y.mean(),
        })
        placements.append(np.bincount(pi * len(ORIGINS) + origin_idx, minlength=len(STATES) * len(ORIGINS)))

        history.append((people, pi, y, placed_community(people, pi, C, R)))
        logs[-1]["delta_hat"] = estimate_delta(history, truth.c_mean, truth.c_sd)
        N, T, C, R = update_community(N, T, pi, origin_idx, scale)   # feedback

    out = pd.DataFrame(logs)
    if log_placements:
        out = pd.concat([out, pd.DataFrame(np.array(placements).reshape(rounds, -1), columns=PLACEMENT_COLUMNS)], axis=1)
    return out


def place(people, u, scores, P, C, R, N, T, origins, cap, model, truth, planner, lookahead, scale, rng):
    """Placement of one cohort by the learned model, the oracle or at random by quota."""
    if planner == "random":
        return rng.permutation(np.repeat(np.arange(len(cap)), cap))
    plan = scores if planner == "model" else P
    if lookahead == 0:
        return assign_exact(plan, cap)
    if planner == "model":
        predict = lambda C_: model.predict_matrix(people, {"C": C_, "R": R})
    else:
        predict = lambda C_: truth.probability_all_states(people, u, C_, R)
    slope = community_slope(predict, people, C)
    return assign_forward(plan, cap, origins, slope, share_increase(N, T, scale), lookahead)


def placed_community(people, states, C, R):
    """Community shares each person met in the state they were placed in (training data for the model)."""
    origins = people["country_of_origin"].map(ORIGIN_IDX).to_numpy()
    return {"C": C[states, origins], "R": R[states, origins]}


RETRAIN = {"fixed": None, "every_3": 3, "every_round": 1}


COMMUNITY_PARAMS = ['delta', 'kappa', 'delta_female', 'delta_recent']


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------

def _merge(base, override, where=''):
    '''Deep-merge override into a copy of base; unknown keys are an error (typo guard).'''
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key not in out and where != 'truth.params':
            raise KeyError(f'unknown config key: {where + "." if where else ""}{key}')
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value, f'{where}.{key}' if where else key)
        else:
            out[key] = value
    return out


def _nested(dotted, value):
    '''"truth.params.delta", 0.3  ->  {'truth': {'params': {'delta': 0.3}}}'''
    for key in reversed(dotted.split('.')):
        value = {key: value}
    return value


def load_config(path):
    '''Read a yaml config; `base:` names another config (relative path) to start from.'''
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    base = config.pop('base', None)
    if base is None:
        return config
    grid = config.pop('grid', None)
    merged = _merge(load_config(path.parent / base), config)
    if grid is not None:
        merged['grid'] = grid
    return merged


def expand(config):
    '''One config per point of the `grid:` (product over dotted keys); no grid -> [config].'''
    config = copy.deepcopy(config)
    grid = config.pop('grid', None) or {}
    out = []
    for values in itertools.product(*grid.values()):
        point = config
        for key, value in zip(grid, values):
            point = _merge(point, _nested(key, value))
        out.append(point)
    return out


def run_key(config, seed):
    '''Readable prefix + hash of every setting that affects the numbers.'''
    settings = {k: v for k, v in config.items() if k not in ('name', 'seeds', 'results_dir')}
    digest = hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    t = config['truth']
    return (f"{t['form']}_{t['interactions'].replace(' ', '')}_{config['policy']['retrain']}"
            f"_h{config['n_hist']}_s{seed}_{digest}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def make_truth(config):
    t = config['truth']
    return GroundTruth(t['form'], interaction_scale=INTERACTION_SCENARIOS[t['interactions']], **t['params'])


def run_simulation(config, seed, truth=None):
    '''One run: one row per round with run_key, settings, all metrics and the placement log.

    truth  a GroundTruth built from config['truth']; pass it to skip re-calibration across seeds.
    '''
    truth = truth or make_truth(config)
    out = simulate(truth, rounds=config['rounds'], cohort_size=config['cohort_size'],
                   n_hist=config['n_hist'], retrain_every=RETRAIN[config['policy']['retrain']],
                   window=config['policy']['window'], scale=config['scale'], seed=seed,
                   log_placements=True, features=config['policy']['features'],
                   planner=config['policy']['planner'], lookahead=config['policy']['lookahead'])
    t = config['truth']
    meta = {'run_key': run_key(config, seed), 'seed': seed, 'form': t['form'],
            'interactions': t['interactions'], 'policy': config['policy']['retrain'],
            'features': config['policy']['features'],
            'planner': config['policy']['planner'], 'lookahead': config['policy']['lookahead'],
            'n_hist': config['n_hist'], 'cohort_size': config['cohort_size'],
            **{p: t['params'].get(p, 0.0) for p in COMMUNITY_PARAMS}}
    for i, (key, value) in enumerate(meta.items()):
        out.insert(i, key, value)
    return out


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

def run_experiment(path):
    '''Run every grid point x seed of a config file; one csv per run in results_dir/name/.

    Finished runs (csv exists) are skipped, so an interrupted experiment can be restarted.
    '''
    config = load_config(path)
    folder = Path(config['results_dir']) / config['name']
    folder.mkdir(parents=True, exist_ok=True)
    truths, start = {}, time.time()
    for point in expand(config):
        tkey = json.dumps(point['truth'], sort_keys=True)
        for seed in point['seeds']:
            key = run_key(point, seed)
            if (folder / f'{key}.csv').exists():
                continue
            if tkey not in truths:
                truths[tkey] = make_truth(point)
            run_simulation(point, seed, truths[tkey]).to_csv(folder / f'{key}.csv', index=False)
            (folder / f'{key}.yaml').write_text(yaml.safe_dump({**point, 'seed': seed}, sort_keys=False))
            print(f'{time.time() - start:6.0f}s  {key}', flush=True)
    return folder


def load_results(folder):
    '''All run csvs of one experiment folder as one table.'''
    return pd.concat([pd.read_csv(f) for f in sorted(Path(folder).glob('*.csv'))], ignore_index=True)


if __name__ == '__main__':
    for p in sys.argv[1:]:
        run_experiment(p)
