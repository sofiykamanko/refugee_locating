"""Policy model (gradient boosting) and capacity-constrained matching.

Moved from refugee_experiments_v2.ipynb, section 3.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier

from .generator import ORIGINS, RELIGION, STATES

ORIGIN_IDX = {o: i for i, o in enumerate(ORIGINS)}
EDU_IDX = {"low": 0, "medium": 1, "high": 2}
GER_IDX = {"none": 0, "poorly": 1, "fairly": 2, "good": 3}
REL_IDX = {r: i for i, r in enumerate(RELIGION)}


USE_RELIGION = False   # 11.11 used religion as a feature; switched off by default (not in the ground truth)


def encode(people):
    """Observable attributes as a numeric matrix. Columns 2 (origin) and 8 (religion) are categorical."""
    religion = people["religion"].map(REL_IDX) if USE_RELIGION else np.zeros(len(people))
    return np.column_stack([
        (people["sex"] == "male").astype(int),
        people["age"],
        people["country_of_origin"].map(ORIGIN_IDX),
        people["education_level"].map(EDU_IDX),
        people["school_years"],
        people["vocational_training"].astype(int),
        people["german_speaking"].map(GER_IDX),
        people["free_case"].astype(int),
        religion,
    ]).astype(float)


class PolicyModel:
    """P(employed | X, state), trained only on observed (X, state, outcome)."""

    def __init__(self, seed=0):
        # column 9 (state) is appended to encode(); categorical = origin, religion, state
        self.clf = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
            categorical_features=[2, 8, 9], early_stopping=False, random_state=seed)

    def fit(self, people, states, y):
        self.clf.fit(np.column_stack([encode(people), states]), y)
        return self

    def predict_matrix(self, people):
        """n x 16 matrix of predicted employment probabilities."""
        # CHANGED (speed): one predict_proba call on all 16 x n rows instead of 16 calls
        base = encode(people)
        n, K = len(base), len(STATES)
        stacked = np.column_stack([np.tile(base, (K, 1)), np.repeat(np.arange(K), n)])
        return self.clf.predict_proba(stacked)[:, 1].reshape(K, n).T


def assign_exact_scipy(scores, cap):
    """27.10 version, kept for the check below: n x n slot matrix, cost grows ~ n^3."""
    slots = np.repeat(np.arange(len(cap)), cap)
    rows, cols = linear_sum_assignment(-scores[:, slots])
    out = np.empty(scores.shape[0], dtype=int)
    out[rows] = slots[cols]
    return out


# CHANGED (speed): exact transportation solver on the n x 16 problem (successive shortest paths)
def assign_fast(scores, cap):
    scores = np.asarray(scores, float)
    n, K = scores.shape
    cap = np.asarray(cap)
    assert cap.sum() == n
    idx = np.arange(n)
    a = scores.argmax(axis=1)                   
    counts = np.bincount(a, minlength=K)
    excess = counts - cap

    def row(s):
        """Cheapest loss of moving one person currently in state s to each state t, and who."""
        members = np.flatnonzero(a == s)
        if members.size == 0:
            return np.full(K, np.inf), np.full(K, -1)
        block = scores[members, s][:, None] - scores[members]   # |s| x K
        j = block.argmin(axis=0)
        w = block[j, np.arange(K)]
        w[s] = np.inf
        return w, members[j]

    W = np.empty((K, K)); who = np.empty((K, K), dtype=int)
    for s in range(K):
        W[s], who[s] = row(s)
    ar = np.arange(K)

    while True:
        over = np.flatnonzero(excess > 0)
        if over.size == 0:
            break
        under = np.flatnonzero(excess < 0)
        # Bellman-Ford from all overfull states (no negative cycles by construction)
        dist = np.full(K, np.inf)
        dist[over] = 0.0
        pred = np.full(K, -1)
        for _ in range(K):
            cand = dist[:, None] + W
            best_s = cand.argmin(axis=0)
            best = cand[best_s, ar]
            improve = best < dist - 1e-15
            if not improve.any():
                break
            dist[improve] = best[improve]
            pred[improve] = best_s[improve]
        sink = under[dist[under].argmin()]
        hops, t = [], sink
        while pred[t] != -1:
            hops.append((pred[t], t))
            t = pred[t]
        src = t
        touched = set()
        for s_, t_ in hops:
            a[who[s_, t_]] = t_
            touched.update((s_, t_))
        excess[src] -= 1
        excess[sink] += 1
        for s_ in touched:                 
            W[s_], who[s_] = row(s_)
    return a


def assign_exact(scores, cap):
    """Maximum-weight assignment of n people to states with capacities cap (same optimum as the scipy version)."""
    return assign_fast(scores, cap)
