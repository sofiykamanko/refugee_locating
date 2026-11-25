"""Can the community effect be learned from the algorithm's own data? (workstream A, step 5)

An analyst who suspects a linear community effect fits a logit on all observed placements:
    employed ~ personal characteristics + state + origin + z
where z is the co-national share each person met when placed, standardised with the moments
of the starting community matrix (observable from Destatis). The coefficient of z estimates delta.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .generator import ORIGIN_IDX, STATES, outcome_features


def estimate_delta(history, c_mean, c_sd):
    """history: list of (people, states, y, {"C": share met, ...}) as kept by simulate()."""
    people = pd.concat([h[0] for h in history], ignore_index=True)
    states = np.concatenate([h[1] for h in history])
    y = np.concatenate([h[2] for h in history])
    z = (np.concatenate([h[3]["C"] for h in history]) - c_mean) / c_sd
    origins = people["country_of_origin"].map(ORIGIN_IDX).to_numpy()
    X = np.column_stack([
        outcome_features(people).to_numpy(),
        np.eye(len(STATES))[states][:, 1:],                  # state dummies
        np.eye(len(ORIGIN_IDX))[origins][:, 1:],             # origin dummies
        z,
    ])
    return LogisticRegression(C=np.inf, max_iter=2000).fit(X, y).coef_[0, -1]
