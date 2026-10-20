"""
Condition A: static null world (Bastani et al. 2026, sec. 5).

For each replication: draw training data under random assignment, fit a model,
optimise the assignment of a fresh test cohort, then evaluate the policy by
 (1) the same model (model-based), (2) bootstrap-refitted models, (3) IPW,
 (4) the TRUE outcome model (only possible in simulation).

Usage: python run_static.py --setting US --model gbm --reps 50 --boot 0
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import CONFIGS, build_config
from env import World, sample_covariates, assign, ipw_rate
from models import MODELS


def draw_data(cfg, world, n, rng, force_free=False):
    df = sample_covariates(cfg, n, rng, force_free=force_free)
    T = rng.choice(cfg["n_locations"], size=n, p=cfg["loc_p"])
    P = world.prob_matrix(df)
    Y = (rng.random(n) < P[np.arange(n), T]).astype(int)
    return df, T, Y, P


def run(setting="US", model="gbm", reps=50, boot=0, seed=0, synergy_sd=0.0, verbose=True):
    cfg = build_config(CONFIGS[setting])
    rng = np.random.default_rng(seed)
    world = World(cfg, rng, synergy_sd=synergy_sd)
    Model = MODELS[model]
    rows = []
    t0 = time.time()
    for r in range(reps):
        dtr, Ttr, Ytr, _ = draw_data(cfg, world, cfg["n_train"], rng)
        m = Model(cfg, seed=r).fit(dtr, Ttr, Ytr)

        dte, Tte, Yte, Pte = draw_data(cfg, world, cfg["n_test"], rng, force_free=cfg["test_only_free"])
        cap = np.bincount(Tte, minlength=cfg["n_locations"])
        pred = m.predict_matrix(dte)
        pi = assign(pred, cap)
        idx = np.arange(len(pi))

        rec = dict(
            rep=r,
            observed=Yte.mean(),                 # observed employment under historical assignment
            true_status_quo=Pte[idx, Tte].mean(),
            true_policy=Pte[idx, pi].mean(),     # ground truth for the learned policy
            model_based=pred[idx, pi].mean(),    # same model evaluates its own policy
            ipw=ipw_rate(pi, Tte, Yte, cfg["loc_p"]),
            train_rate=Ytr.mean(),
            auc=roc_auc_score(Yte, pred[idx, Tte]),
        )
        if boot:
            vals = []
            for b in range(boot):
                bi = rng.integers(0, len(Ttr), len(Ttr))
                mb = Model(cfg, seed=1000 + b).fit(dtr.iloc[bi].reset_index(drop=True), Ttr[bi], Ytr[bi])
                vals.append(mb.predict_matrix(dte)[idx, pi].mean())
            rec["bootstrap_model_based"] = float(np.mean(vals))
        rows.append(rec)
        if verbose and (r + 1) % max(1, reps // 10) == 0:
            print(f"  rep {r + 1}/{reps}  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    # % change relative to observed employment in the test set (as in Bastani Fig. 2)
    for c in ["true_policy", "model_based", "ipw", "bootstrap_model_based"]:
        if c in df:
            df[f"gain_{c}_pct"] = 100 * (df[c] / df["observed"] - 1)
    df.attrs.update(setting=setting, model=model, synergy_sd=synergy_sd)
    return df


def summarise(df):
    out = {"setting": df.attrs["setting"], "model": df.attrs["model"],
           "train_rate": round(df["train_rate"].mean(), 3),
           "test_rate": round(df["observed"].mean(), 3),
           "auc": round(df["auc"].mean(), 3)}
    for c in [c for c in df.columns if c.startswith("gain_")]:
        out[c.replace("gain_", "").replace("_pct", "")] = \
            f"{df[c].mean():+.1f}% (sd {df[c].std():.1f})"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="US", choices=list(CONFIGS))
    ap.add_argument("--model", default="gbm", choices=list(MODELS))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synergy_sd", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.setting, a.model, a.reps, a.boot, a.seed, a.synergy_sd)
    print(pd.Series(summarise(res)).to_string())
    if a.out:
        res.to_csv(a.out, index=False)
