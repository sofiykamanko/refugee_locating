"""
Synthetic refugee-matching environment (static), following Bastani et al. (2026) sec. 5.

True employment probability:
    f(x, l) = 0.5 * fX(x) + 0.5 * fL(l) [+ synergy(origin, l)]
With synergy = 0 there is no x-by-location interaction, so (at full capacity)
no assignment policy can change expected employment -> any estimated gain is spurious.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import expit


# ----------------------------------------------------------------------------
# Covariates
# ----------------------------------------------------------------------------
def sample_covariates(cfg, n, rng, force_free=False):
    bins = cfg["age_bins"]
    bi = rng.choice(len(bins), size=n, p=[b[2] for b in bins])
    lo = np.array([bins[i][0] for i in bi])
    hi = np.array([bins[i][1] for i in bi])
    age = rng.uniform(lo, hi)
    df = pd.DataFrame({
        "age": age,
        "male": rng.random(n) < cfg["p_male"],
        "education": rng.choice(cfg["education"]["levels"], size=n, p=cfg["education"]["p"]),
        "language": rng.random(n) < cfg["p_language"],
        "origin": rng.choice(cfg["origins"], size=n, p=cfg["origin_p"]),
        "year": rng.choice(cfg["years"], size=n, p=cfg["year_p"]),
        "month": rng.choice(12, size=n, p=cfg["month_p"]) + 1,
    })
    if cfg["p_free_case"] is not None:
        df["free_case"] = True if force_free else (rng.random(n) < cfg["p_free_case"])
    return df


def encode(cfg, df):
    """One-hot design matrix with a fixed column set (independent of the sample)."""
    cols = {"age_std": (df["age"].values - 35.0) / 10.0,
            "male": df["male"].astype(float).values,
            "language": df["language"].astype(float).values}
    for lev in cfg["education"]["levels"][1:]:
        cols[f"edu_{lev}"] = (df["education"] == lev).astype(float).values
    for o in cfg["origins"][1:]:
        cols[f"orig_{o}"] = (df["origin"] == o).astype(float).values
    for y in cfg["years"][1:]:
        cols[f"year_{y}"] = (df["year"] == y).astype(float).values
    for m in range(2, 13):
        cols[f"month_{m}"] = (df["month"] == m).astype(float).values
    if "free_case" in df:
        cols["free_case"] = df["free_case"].astype(float).values
    return pd.DataFrame(cols)


# ----------------------------------------------------------------------------
# Outcome model
# ----------------------------------------------------------------------------
class World:
    """Holds the ground-truth outcome model for one simulation run."""

    def __init__(self, cfg, rng, coef_sd=0.6, free_bonus=None, synergy_sd=0.0):
        self.cfg = cfg
        L = cfg["n_locations"]
        p = cfg["loc_p"]

        # --- location effect fL
        if cfg["loc_effect"] == "beta12":
            for _ in range(100):
                fL = rng.beta(1, 2, size=L)
                mL = float(p @ fL)
                need_all = 2 * cfg["target_rate_all"] - mL
                need_free = 2 * cfg["target_rate_free"] - mL if cfg["target_rate_free"] else 0.5
                if abs(mL - 1 / 3) < 0.03 and 0.02 < need_all < need_free < 0.95:
                    break
            self.fL = fL
        elif cfg["loc_effect"] == "proportional":
            lo, hi = cfg["loc_rel_spread"]
            rel = rng.uniform(lo, hi, size=L)
            rel = rel / rel.max()
            # Land-level expected rates proportional to rel, averaging to the target:
            #   rate_l = 0.5*mean(fX) + 0.5*fL_l ; we set mean(fX) = target.
            t = cfg["target_rate_all"]
            rate = rel * t / float(p @ rel)
            self.fL = 2 * rate - t
        else:
            raise ValueError(cfg["loc_effect"])
        mL = float(p @ self.fL)

        # --- person effect fX: logit with random coefficients
        ref = sample_covariates(cfg, 20000, rng)
        Xref = encode(cfg, ref)
        self.columns = list(Xref.columns)
        w = rng.normal(0, coef_sd, size=len(self.columns))
        if "free_case" in self.columns:
            w[self.columns.index("free_case")] = 0.0
        self.w = pd.Series(w, index=self.columns)

        target_fx_all = 2 * cfg["target_rate_all"] - mL
        if "free_case" in self.columns and cfg["target_rate_free"]:
            target_fx_free = 2 * cfg["target_rate_free"] - mL
            is_free = ref["free_case"].values
            lo_b, hi_b = 0.0, 10.0
            for _ in range(40):
                b = 0.5 * (lo_b + hi_b)
                self.w["free_case"] = b
                self._calibrate_intercept(Xref, target_fx_all)
                m_free = self.fX_from_design(Xref)[is_free].mean()
                if m_free < target_fx_free:
                    lo_b = b
                else:
                    hi_b = b
        else:
            self._calibrate_intercept(Xref, target_fx_all)

        # --- optional true origin x location synergy (world "B": matching can help)
        O = len(cfg["origins"])
        self.synergy = rng.normal(0, synergy_sd, size=(O, L)) if synergy_sd > 0 else np.zeros((O, L))
        self.origin_idx = {o: i for i, o in enumerate(cfg["origins"])}

    def _calibrate_intercept(self, X, target):
        lo, hi = -10.0, 10.0
        for _ in range(50):
            self.b0 = 0.5 * (lo + hi)
            if self.fX_from_design(X).mean() < target:
                lo = self.b0
            else:
                hi = self.b0

    def fX_from_design(self, X):
        return expit(self.b0 + X[self.columns].values @ self.w.values)

    def fX(self, df):
        return self.fX_from_design(encode(self.cfg, df))

    def prob_matrix(self, df, synergy=None):
        """n x L matrix of true employment probabilities."""
        syn = self.synergy if synergy is None else synergy
        fx = self.fX(df)[:, None]
        oi = df["origin"].map(self.origin_idx).values
        P = 0.5 * fx + 0.5 * self.fL[None, :] + syn[oi, :]
        return np.clip(P, 0.001, 0.999)


# ----------------------------------------------------------------------------
# Assignment: maximise predicted employment subject to location capacities
# ----------------------------------------------------------------------------
def assign(score, capacity):
    """score: n x L; capacity: length-L ints summing to >= n. Exact via expanded LSAP."""
    n, L = score.shape
    slots = np.repeat(np.arange(L), capacity)
    cost = -score[:, slots]
    rows, cols = linear_sum_assignment(cost)
    out = np.empty(n, dtype=int)
    out[rows] = slots[cols]
    return out


def ipw_rate(pi, T, Y, p):
    """Horvitz-Thompson estimate of the employment rate under policy pi."""
    match = pi == T
    return float(np.sum(Y[match] / p[T[match]]) / len(Y))
