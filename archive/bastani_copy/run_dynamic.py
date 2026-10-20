"""
Dynamic simulation: does a welfare-improving matching policy erode its own gains?

World "B" (static benefit): true origin x location synergy exists, so matching can help.
Each round a new cohort arrives, the current model assigns it under Land capacities,
outcomes are realised, the model is retrained on the growing data pool.

Conditions
  B  static   : synergy never changes                          (kappa=0, kappa_exo=0)
  C  endogenous: positive synergy of (origin, Land) erodes with the EXCESS stock of that
                 origin placed there by the policy, relative to what random quota
                 assignment would have placed                   (kappa>0)
  D  exogenous : synergy erodes at a fixed rate each round, independent of the policy
                 (kappa_exo>0) -> control of matched magnitude

Evaluation per round (all in the SAME round, i.e. under the current state of the world):
  true_policy    : ground-truth employment under the learned assignment
  true_random    : ground-truth expected employment under random quota assignment
  model_promised : what the model predicts for its own assignment (model-based)
  ipw            : IPW estimate from the epsilon-exploration subset
Usage: python run_dynamic.py --rounds 8 --seeds 5
"""
import argparse
import time

import numpy as np
import pandas as pd

from config import CONFIGS, build_config
from env import World, sample_covariates, assign
from models import MODELS


def capacities(n, p):
    cap = np.floor(n * p).astype(int)
    cap[np.argsort(-(n * p - cap))[: n - cap.sum()]] += 1
    return cap


def simulate(cfg, seed, kappa=0.0, kappa_exo=0.0, rounds=8, cohort=2000, n_hist=10000,
             synergy_sd=0.04, eps=0.1, model="gbm", retrain_every=1, window=None):
    rng = np.random.default_rng(seed)
    world = World(cfg, rng, synergy_sd=synergy_sd)
    L, O = cfg["n_locations"], len(cfg["origins"])
    p = cfg["loc_p"]
    base_syn = world.synergy.copy()
    excess = np.zeros((O, L))          # excess placements vs. random-quota expectation
    exp_cell = cohort * np.outer(cfg["origin_p"], p)   # expected count per round under random

    def current_synergy(r):
        pos = base_syn > 0
        s = base_syn.copy()
        s[pos] *= np.exp(-kappa * np.maximum(excess[pos], 0) / exp_cell[pos])
        s *= np.exp(-kappa_exo * r)
        return s

    # historical data under random assignment
    d0 = sample_covariates(cfg, n_hist, rng)
    T0 = rng.choice(L, size=n_hist, p=p)
    P0 = world.prob_matrix(d0, current_synergy(0))
    Y0 = (rng.random(n_hist) < P0[np.arange(n_hist), T0]).astype(int)
    pool = [(d0, T0, Y0, 0)]

    rows, m = [], None
    cap = capacities(cohort, p)
    for r in range(1, rounds + 1):
        if m is None or (r - 1) % retrain_every == 0:
            use = pool if window is None else [x for x in pool if x[3] >= r - window]
            D = pd.concat([x[0] for x in use], ignore_index=True)
            T = np.concatenate([x[1] for x in use]); Y = np.concatenate([x[2] for x in use])
            m = MODELS[model](cfg, seed=seed * 100 + r).fit(D, T, Y)

        syn = current_synergy(r)
        d = sample_covariates(cfg, cohort, rng)
        Ptrue = world.prob_matrix(d, syn)
        pred = m.predict_matrix(d)
        pi = assign(pred, cap)

        # epsilon exploration with known propensities (keeps IPW defined)
        explore = rng.random(cohort) < eps
        T_act = pi.copy()
        T_act[explore] = rng.choice(L, size=explore.sum(), p=p)
        idx = np.arange(cohort)
        Y = (rng.random(cohort) < Ptrue[idx, T_act]).astype(int)

        e = explore
        ipw = float(np.sum(((pi[e] == T_act[e]) * Y[e]) / p[T_act[e]]) / e.sum())
        rows.append(dict(
            seed=seed, round=r,
            true_policy=Ptrue[idx, pi].mean(),
            true_random=(Ptrue @ p).mean(),
            model_promised=pred[idx, pi].mean(),
            ipw=ipw,
            mean_pos_synergy=float(syn[base_syn > 0].mean()),
        ))

        # update world state with realised placements
        oi = d["origin"].map(world.origin_idx).values
        placed = np.zeros((O, L))
        np.add.at(placed, (oi, T_act), 1)
        excess += placed - exp_cell
        pool.append((d, T_act, Y, r))
    return rows


CONDITIONS = {
    "B_static": dict(kappa=0.0, kappa_exo=0.0),
    "C_endogenous": dict(kappa=0.3, kappa_exo=0.0),
    "D_exogenous": dict(kappa=0.0, kappa_exo=0.12)  # matched to C: similar synergy decay by round 8,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="DE")
    ap.add_argument("--model", default="gbm")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results/dynamic_DE.csv")
    a = ap.parse_args()
    cfg = build_config(CONFIGS[a.setting])
    allrows, t0 = [], time.time()
    for name, kw in CONDITIONS.items():
        for s in range(a.seeds):
            for row in simulate(cfg, s, rounds=a.rounds, model=a.model, **kw):
                row["condition"] = name
                allrows.append(row)
            print(f"{name} seed {s} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(allrows)
    for c in ["true_policy", "model_promised", "ipw"]:
        df[f"gain_{c}_pct"] = 100 * (df[c] / df["true_random"] - 1)
    df.to_csv(a.out, index=False)
    print(df.groupby(["condition", "round"])[["gain_true_policy_pct", "gain_model_promised_pct",
                                               "gain_ipw_pct", "mean_pos_synergy"]].mean().round(2))
