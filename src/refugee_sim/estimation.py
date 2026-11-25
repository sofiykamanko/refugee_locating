"""Can the community effect be learned from the algorithm's own data? (workstream A, step 5)

An analyst who suspects a linear community effect fits a logit on all observed placements:
    employed ~ personal characteristics + state + origin + z
where z is the co-national share each person met when placed, standardised with the moments
of the starting community matrix (observable from Destatis). The coefficient of z estimates delta.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .generator import ORIGIN_IDX, STATES, initial_community_matrix, outcome_features

C0 = initial_community_matrix()
C_MEAN, C_SD = C0.mean(), C0.std()


def standardise(C):
    return (C - C_MEAN) / C_SD


def fit_delta(people, states, y, C_placed):
    """Logit coefficient of z, with characteristics, state and origin as controls."""
    origins = people["country_of_origin"].map(ORIGIN_IDX).to_numpy()
    X = np.column_stack([
        outcome_features(people).to_numpy(),
        np.eye(len(STATES))[states][:, 1:],                  # state dummies
        np.eye(len(ORIGIN_IDX))[origins][:, 1:],             # origin dummies
        standardise(C_placed),
    ])
    return LogisticRegression(C=np.inf, max_iter=2000).fit(X, y).coef_[0, -1]


def estimate_delta(history):
    """history: list of (people, states, y, {"C": share met, ...}) as kept by simulate()."""
    return fit_delta(pd.concat([h[0] for h in history], ignore_index=True),
                     np.concatenate([h[1] for h in history]),
                     np.concatenate([h[2] for h in history]),
                     np.concatenate([h[3]["C"] for h in history]))
