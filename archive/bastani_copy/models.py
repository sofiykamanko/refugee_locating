"""Outcome models that predict P(employed | x, location) for every location."""
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from env import encode


class PerLocationGBM:
    """One gradient-boosted classifier per location (as in Bansak et al. 2018)."""

    def __init__(self, cfg, seed=0, **kw):
        self.cfg = cfg
        self.seed = seed
        # tuned so that the US null world gives a spurious gain near Bastani's ~59%
        self.kw = dict(max_iter=100, learning_rate=0.05, max_depth=3, min_samples_leaf=30, **kw)

    def fit(self, df, T, Y):
        X = encode(self.cfg, df).values
        self.models, self.const = {}, {}
        for l in range(self.cfg["n_locations"]):
            m = T == l
            if m.sum() < 20 or Y[m].min() == Y[m].max():
                self.const[l] = float(Y[m].mean()) if m.any() else float(Y.mean())
                continue
            clf = HistGradientBoostingClassifier(random_state=self.seed, **self.kw)
            clf.fit(X[m], Y[m])
            self.models[l] = clf
        return self

    def predict_matrix(self, df):
        X = encode(self.cfg, df).values
        L = self.cfg["n_locations"]
        out = np.empty((len(df), L))
        for l in range(L):
            out[:, l] = self.models[l].predict_proba(X)[:, 1] if l in self.models else self.const[l]
        return out


class LassoInteractions:
    """Pooled L1-logit: covariates + location dummies + (selected covariates x location)."""

    INTERACT = ("age_std", "male", "language", "edu_", "orig_")

    def __init__(self, cfg, C=0.2, seed=0):  # C tuned: US null world -> ~13% (Bastani)
        self.cfg, self.C, self.seed = cfg, C, seed

    def _design(self, Xc, T):
        n, L = len(T), self.cfg["n_locations"]
        D = sp.csr_matrix((np.ones(n), (np.arange(n), T)), shape=(n, L))
        icols = [c for c in Xc.columns if c.startswith(self.INTERACT)]
        blocks = [sp.csr_matrix(Xc.values), D]
        for c in icols:
            blocks.append(D.multiply(Xc[c].values[:, None]))
        return sp.hstack(blocks).tocsr()

    def fit(self, df, T, Y):
        Xc = encode(self.cfg, df)
        self.clf = LogisticRegression(penalty="l1", C=self.C, solver="liblinear", max_iter=500)
        self.clf.fit(self._design(Xc, T), Y)
        return self

    def predict_matrix(self, df):
        Xc = encode(self.cfg, df)
        n, L = len(df), self.cfg["n_locations"]
        out = np.empty((n, L))
        for l in range(L):
            out[:, l] = self.clf.predict_proba(self._design(Xc, np.full(n, l)))[:, 1]
        return out


MODELS = {"gbm": PerLocationGBM, "lasso": LassoInteractions}
