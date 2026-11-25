"""Synthetic population and ground-truth outcome model (v2).

Moved from refugee_experiments_v2.ipynb, sections 1 and 2. generate() and the cli at the
bottom come from the old scripts/synthetic_data_generator.py.

    python -m refugee_sim.generator --n 5000 --seed 42 --out data/synthetic/synthetic_refugees.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm, truncnorm


# Marginal distributions (Strasser Ceballos & Kern 2025, Table 5)
P_MALE = 0.602
P_VOCATIONAL_TRAINING = 0.248
P_FREE_CASE = 0.539

AGE = {"mean": 32.0, "sd": 10.0, "low": 18, "high": 65}
SCHOOL_YEARS = {"mean": 9.9, "sd": 3.0, "low": 0, "high": 18}

COUNTRY_OF_ORIGIN = {
    "Syria": 52.8, "Iraq": 13.3, "Afghanistan": 12.5, "Eritrea": 4.3, "Iran": 2.3,
    "Pakistan": 1.7, "Russian Fed.": 1.5, "Somalia": 1.2, "Other": 10.4,
}
EDUCATION_LEVEL = {"low": 33.9, "medium": 26.8, "high": 39.3}
GERMAN_SPEAKING = {"none": 93.9, "poorly": 2.9, "fairly": 1.4, "good": 1.8}
RELIGION = {
    "Shiite": 7.6, "Sunnite": 60.6, "Other Islamic": 6.7, "Other Christian": 6.7,
    "No religion": 5.8, "Other": 6.2, "Protestant": 2.7, "Catholic": 2.5,
    "Christian Orthodox": 1.0,
}

# Königsteiner Schlüssel, current EASY quotas (BAMF, "Erstverteilung")
STATE_QUOTA = {
    "NW": 20.96090, "BY": 15.68563, "BW": 13.12509, "NI": 9.40263,
    "HE": 7.40812, "BE": 5.20075, "SN": 4.94788, "RP": 4.80876,
    "SH": 3.39588, "BB": 3.01025, "ST": 2.69612, "HH": 2.63402,
    "TH": 2.61509, "MV": 1.96964, "SL": 1.18749, "HB": 0.95455,
}

# Foreigners by state and nationality (Destatis GENESIS 12521-0021, 31 Dec 2025).
# Order: Syria, Iraq, Afghanistan, Eritrea, Iran, Pakistan, Russian Fed., Somalia,
#        all foreigners, average count per nationality outside these eight ("Other").
DESTATIS_COUNTS = {
    "NW": [280650, 87800, 75265, 16330, 44310, 16185, 54065, 12350, 3265705, 14099],
    "BY": [88860, 31755, 61580, 12405, 16095, 13755, 43305, 11460, 2386525, 11269],
    "BW": [93815, 29210, 44075, 9465, 13985, 13215, 33215, 5635, 2192400, 10483],
    "NI": [97685, 44760, 32700, 4080, 13150, 6065, 21645, 4560, 1084595, 4804],
    "HE": [59430, 10745, 69515, 18975, 18155, 21240, 20940, 13885, 1360875, 6164],
    "BE": [41455, 9970, 25070, 1950, 13375, 8850, 35165, 1340, 994590, 4635],
    "SN": [36320, 6255, 15035, 2065, 4350, 4605, 14005, 1200, 371045, 1710],
    "RP": [48700, 3980, 21310, 4025, 6400, 7070, 10320, 7135, 626355, 2991],
    "SH": [36225, 12390, 23380, 3550, 6050, 1590, 8660, 1670, 352575, 1570],
    "BB": [19435, 1530, 13675, 1335, 3330, 2770, 11505, 1515, 213330, 977],
    "ST": [29165, 2335, 9240, 1925, 2885, 1460, 4770, 1325, 192155, 897],
    "HH": [16800, 4540, 33060, 3650, 12250, 2055, 10620, 2020, 387845, 1731],
    "TH": [22280, 4265, 11480, 1890, 2070, 1570, 5120, 950, 187990, 929],
    "MV": [12685, 1375, 6520, 885, 1820, 470, 4190, 855, 125710, 659],
    "SL": [34640, 1215, 2145, 1180, 800, 785, 2410, 100, 164395, 781],
    "HB": [18135, 1930, 5725, 1025, 2765, 930, 4290, 2045, 164135, 816],
}


# ---------------------------------------------------------------------------
# Individual effects beta (log-odds scale)
# ---------------------------------------------------------------------------
# Source: Brücker, Jaschke & Kosyakova (2025), IAB-Forum "10 Jahre Fluchtmigration 2015",
# average marginal effects (AME) on the probability of paid employment, IAB-BAMF-SOEP survey
# linked to BA social-security records. They are associations, not causal effects.
#   medium school education  +3 pp   (reference: primary or no school)
#   higher school education  +5 pp, university +6 pp  -> our "high" uses the midpoint, +5.5 pp
#   good / very good German  +7 pp   (reference: all lower levels)
#   age 31-50                 ~0 pp, age 51-64  -14.5 pp   (reference: 18-30)
# AMEs are converted with beta = AME / (p (1 - p)); p = 0.4 approximates the mean employment
# rate in that sample (our approximation, not reported in the source).
AME_TO_LOGIT = 1 / (0.4 * 0.6)

BETA = {
    "male": 1.7,                                # SOEP estimate (Kern, n = 5,400)
    "age_51_plus": -1.6,                        # SOEP estimate (IAB-Forum 2025 reports -14.5 pp)
    "education_medium": 0.030 * AME_TO_LOGIT,   # IAB-Forum 2025, close to SOEP
    "education_high": 0.055 * AME_TO_LOGIT,     # IAB-Forum 2025, close to SOEP
    "german_good": 0.070 * AME_TO_LOGIT,        # IAB-Forum 2025, close to SOEP
    "vocational_training": 0.12,                # close to SOEP
    "free_case": 0.36,                          # SOEP estimate
}
# School years carry no separate effect (linked to education through the copula).
# Religion is generated but is not part of the outcome equation.


# Characteristic x state interactions (log-odds), from a random-slopes logistic regression on the
# IAB-BAMF-SOEP sample (Kern): for each characteristic, how much larger or smaller its effect is in
# each state than on average (quota-weighted mean 0). None is statistically significant.
# Order of the values: NW, BY, BW, NI, HE, BE, SN, RP, SH, BB, ST, HH, TH, MV, SL, HB
STATE_INTERACTIONS = {
    "male": [0.0651, -0.0087, -0.0298, -0.0434, -0.0517, 0.0023, 0.0079, -0.0094, 0.0283, 0.0111, 0.0096, -0.0285, -0.0272, -0.0057, -0.0346, -0.0082],
    "age_51_plus": [-0.1538, 0.1265, 0.2248, -0.0088, 0.0909, -0.1759, -0.1380, 0.0554, -0.1776, -0.1483, -0.0754, 0.0255, 0.0683, -0.0221, 0.0973, -0.0372],
    "education_medium": [-0.0722, 0.0650, 0.0798, -0.0113, 0.0117, -0.0602, -0.0004, 0.0127, -0.0934, -0.0055, 0.0207, 0.0214, -0.0082, -0.0084, -0.0045, -0.0149],
    "education_high": [-0.1504, 0.1021, 0.1726, -0.0374, 0.0197, -0.0968, -0.0218, 0.0518, 0.0019, -0.0763, -0.0637, 0.0397, 0.0883, -0.0662, 0.0524, -0.0183],
    "german_good": [0.4532, -0.2932, -0.6980, -0.0141, -0.4672, 0.7705, 0.6963, -0.5830, 0.3243, 0.2699, 0.1961, 0.4002, -0.0049, 0.1656, -0.5482, 0.0482],
    "vocational_training": [-0.1424, -0.0066, 0.0616, -0.0405, -0.0642, 0.1102, 0.0911, 0.0608, 0.1940, 0.1006, 0.0168, 0.1138, 0.1302, 0.0352, 0.0620, 0.0329],
    "free_case": [0.0991, -0.0949, -0.2105, 0.0092, -0.0545, 0.1927, 0.1115, -0.0360, 0.2196, 0.0939, 0.0516, 0.0361, -0.0468, 0.0019, -0.0917, 0.0796],
}
# Scenarios: 1 = SOEP values (realistic), 4 = stylised, interactions four times as large (assumption).
INTERACTION_SCENARIOS = {"none": 0.0, "SOEP": 1.0, "stylised x4": 4.0}

SIGMA_U = 0.5   # sd of unobserved individual differences u_i (placeholder)

# Employment rate of nationals of the eight main asylum countries by state, January 2026
# (Bundesagentur für Arbeit). Only relative differences between states are used.
STATE_EMPLOYMENT_RATE_BA = {
    "NW": 49.2, "BY": 55.3, "BW": 52.7, "NI": 48.0, "HE": 48.4, "BE": 49.9, "SN": 46.4, "RP": 49.6,
    "SH": 48.5, "BB": 44.2, "ST": 44.1, "HH": 54.8, "TH": 50.1, "MV": 44.6, "SL": 45.1, "HB": 46.3,
}

# Outcome: employed within three years of arrival.
TARGETS = {
    "employment_rate": 0.15,        # Strasser Ceballos & Kern (2025); SOEP 2014-2019 arrivals: 22.8%
    "state_relative_level": {s: r / max(STATE_EMPLOYMENT_RATE_BA.values())
                             for s, r in STATE_EMPLOYMENT_RATE_BA.items()},
}

EDUCATION_SCHOOL_CORRELATION = 0.6

RELIGION_BY_ORIGIN_SHAPE = {
    #               Shiite Sunnite OthIsl OthChr NoRel Other Prot  Cath  Orth
    "Syria":        [5,    70,     8,     5,     4,    5,    1,    1,    1],
    "Iraq":         [45,   25,     3,     8,     3,    13,   1,    1,    1],
    "Afghanistan":  [20,   65,     3,     2,     5,    3,    1,    0.5,  0.5],
    "Eritrea":      [1,    30,     2,     15,    2,    2,    5,    5,    38],
    "Iran":         [45,   5,      3,     15,    15,   10,   4,    1,    2],
    "Pakistan":     [20,   70,     5,     2,     1,    1,    0.5,  0.5,  0.5],
    "Russian Fed.": [1,    70,     3,     2,     5,    3,    1,    1,    14],
    "Somalia":      [1,    92,     3,     1,     1,    1,    0.5,  0.5,  0.5],
    "Other":        [8,    30,     8,     12,    12,   10,   6,    6,    8],
}

COLUMNS = [
    "refugee_id", "sex", "age", "country_of_origin", "education_level", "school_years",
    "vocational_training", "german_speaking", "free_case", "religion",
    "assigned_state", "true_employment_prob", "employed",
]
STATES = list(STATE_QUOTA)
ORIGINS = list(COUNTRY_OF_ORIGIN)
ORIGIN_IDX = {o: i for i, o in enumerate(ORIGINS)}


def normalise(weights):
    """Percentages (dict or list) -> probability vector that sums to 1."""
    values = np.asarray(list(weights.values()) if isinstance(weights, dict) else weights, float)
    return values / values.sum()


def inverse_transform_sample(rng, distribution, n):
    """Draw n categories from {level: weight} by inverse transform sampling."""
    return categories_from_uniform(rng.random(n), distribution)


def bernoulli(rng, p, n):
    return rng.random(n) < p


def truncated_normal_params(mean, sd, low, high):
    """Shape parameters of a truncated normal on [low, high] whose mean after truncation is `mean`."""
    loc = mean
    for _ in range(50):
        a, b = (low - loc) / sd, (high - loc) / sd
        loc += mean - truncnorm.mean(a, b, loc=loc, scale=sd)
    return (low - loc) / sd, (high - loc) / sd, loc, sd


def truncated_normal(rng, n, mean, sd, low, high):
    a, b, loc, scale = truncated_normal_params(mean, sd, low, high)
    return truncnorm.rvs(a, b, loc=loc, scale=scale, size=n, random_state=rng)


def truncated_normal_from_uniform(u, mean, sd, low, high):
    """Same distribution as truncated_normal, but driven by given uniform numbers u."""
    a, b, loc, scale = truncated_normal_params(mean, sd, low, high)
    return truncnorm.ppf(u, a, b, loc=loc, scale=scale)


def categories_from_uniform(u, distribution):
    """Inverse transform: map uniform numbers u to categories with the given weights."""
    levels = np.array(list(distribution))
    cdf = np.cumsum(normalise(distribution))
    cdf[-1] = 1.0
    return levels[np.searchsorted(cdf, u, side="right")]


def iterative_proportional_fitting(shape, row_targets, column_targets, iterations=500, tol=1e-12):
    """Rescale a positive matrix until its row and column sums match the targets."""
    table = np.asarray(shape, float).copy()
    for _ in range(iterations):
        table *= (row_targets / table.sum(axis=1))[:, None]
        table *= (column_targets / table.sum(axis=0))[None, :]
        if np.abs(table.sum(axis=1) - row_targets).max() < tol:
            break
    return table


def religion_given_origin():
    """P(religion | origin): rows sum to 1, implied marginals match Table 5."""
    shape = [RELIGION_BY_ORIGIN_SHAPE[o] for o in ORIGINS]
    joint = iterative_proportional_fitting(shape, normalise(COUNTRY_OF_ORIGIN), normalise(RELIGION))
    return joint / joint.sum(axis=1, keepdims=True)


def initial_community_matrix():
    """C_0 (16 x 9): share of each origin among all foreigners in each state."""
    labels = ORIGINS[:-1] + ["all_foreigners", "Other"]
    counts = pd.DataFrame(DESTATIS_COUNTS, index=labels).T
    return counts[ORIGINS].div(counts["all_foreigners"], axis=0).loc[STATES].to_numpy()


def sample_attributes(n, rng):
    people = pd.DataFrame({
        "sex": np.where(bernoulli(rng, P_MALE, n), "male", "female"),
        "age": truncated_normal(rng, n, **AGE).round(1),
        "country_of_origin": inverse_transform_sample(rng, COUNTRY_OF_ORIGIN, n),
        "vocational_training": bernoulli(rng, P_VOCATIONAL_TRAINING, n),
        "german_speaking": inverse_transform_sample(rng, GERMAN_SPEAKING, n),
        "free_case": bernoulli(rng, P_FREE_CASE, n),
    })

    # Education level and years of school: Gaussian copula
    rho = EDUCATION_SCHOOL_CORRELATION
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    u = norm.cdf(z)
    people["education_level"] = categories_from_uniform(u[:, 0], EDUCATION_LEVEL)
    people["school_years"] = truncated_normal_from_uniform(u[:, 1], **SCHOOL_YEARS).round(1)

    # Religion drawn conditionally on country of origin
    conditional = religion_given_origin()
    religion = np.empty(n, dtype=object)
    for i, origin in enumerate(ORIGINS):
        members = (people["country_of_origin"] == origin).to_numpy()
        religion[members] = inverse_transform_sample(
            rng, dict(zip(RELIGION, conditional[i])), members.sum())
    people["religion"] = religion

    return people[["sex", "age", "country_of_origin", "education_level", "school_years",
                   "vocational_training", "german_speaking", "free_case", "religion"]]


def capacities(n):
    """c_k = floor(N * q_k); leftover seats go to the largest remainders so that sum(c_k) = N."""
    q = normalise(STATE_QUOTA)
    c = np.floor(n * q).astype(int)
    c[np.argsort(-(n * q - c))[: n - c.sum()]] += 1
    return c


def assign_states(n, rng, mode="random"):
    """Quasi-random allocation by quota ("random" or "exact")."""
    if mode == "random":
        return inverse_transform_sample(rng, dict(zip(range(len(STATES)), STATE_QUOTA.values())), n).astype(int)
    if mode == "exact":
        return rng.permutation(np.repeat(np.arange(len(STATES)), capacities(n)))
    raise ValueError("mode must be 'random' or 'exact'")


def outcome_features(people):
    return pd.DataFrame({
        "male": (people["sex"] == "male").astype(float),
        "age_51_plus": (people["age"] > 50).astype(float),
        "education_medium": (people["education_level"] == "medium").astype(float),
        "education_high": (people["education_level"] == "high").astype(float),
        "german_good": (people["german_speaking"] == "good").astype(float),
        "vocational_training": people["vocational_training"].astype(float),
        "free_case": people["free_case"].astype(float),
    })


# Community effect: how the size of the co-national community changes the log-odds.
#   z   standardised community share (stock), z = (C - mean(C0)) / sd(C0), fixed moments
#   zr  co-nationals who arrived in the PREVIOUS round, in the same sd units (0 at the start)
#   f   1 for women, 0 for men
COMMUNITY_FORMS = {
    # constant effect per sd of community size
    "linear":          lambda z, zr, f, p: p["delta"] * z,
    # small communities help, large ones hurt (peak at z = delta / (2 kappa))
    "inverted_u":      lambda z, zr, f, p: p["delta"] * z - p["kappa"] * z ** 2,
    # diminishing returns: the effect flattens for very large communities
    "saturating":      lambda z, zr, f, p: p["delta"] * np.tanh(z),
    # interaction: the effect is larger for women (Martén et al. 2019 find women benefit most)
    "women_stronger":  lambda z, zr, f, p: (p["delta"] + p["delta_female"] * f) * z,
    # established community helps, newly arrived co-nationals compete (Beaman 2012)
    "stock_vs_recent": lambda z, zr, f, p: p["delta"] * z + p["delta_recent"] * zr,
}
DEFAULT_COMMUNITY_PARAMS = {"delta": 0.0, "kappa": 0.0, "delta_female": 0.0, "delta_recent": 0.0}


class GroundTruth:
    """p_i = sigmoid(alpha + beta'X_i + gamma[T_i] + community(z_i, zr_i, female_i) + u_i)

    form    one of COMMUNITY_FORMS
    params  parameters of that form (see DEFAULT_COMMUNITY_PARAMS)
    alpha, beta["male"] and gamma are calibrated to TARGETS at the starting community C0.
    """

    def __init__(self, form="linear", beta=BETA, sigma_u=SIGMA_U, targets=TARGETS,
                 calibration_seed=12345, calibration_size=50_000, interaction_scale=0.0, **params):
        self.form = form
        self.interaction_scale = interaction_scale
        self.params = {**DEFAULT_COMMUNITY_PARAMS, **params}
        self.beta = dict(beta)
        self.sigma_u = sigma_u
        self.C0 = initial_community_matrix()
        self.c_mean, self.c_sd = self.C0.mean(), self.C0.std()   # fixed baseline moments
        self.alpha = 0.0
        self.gamma = np.zeros(len(STATES))
        self._calibrate(targets, np.random.default_rng(calibration_seed), calibration_size)

    def label(self):
        shown = {k: v for k, v in self.params.items() if v != 0}
        return f"{self.form} {shown}" if shown else f"{self.form} (no effect)"

    def _community(self, people, C, R):
        """n x 16 matrix of community terms."""
        origin = people["country_of_origin"].map(ORIGIN_IDX).to_numpy()
        z = (C[:, origin].T - self.c_mean) / self.c_sd
        zr = R[:, origin].T / self.c_sd
        female = (people["sex"] == "female").to_numpy(float)[:, None]
        return COMMUNITY_FORMS[self.form](z, zr, female, self.params)

    def _individual(self, people, u):
        """n x 16: personal part, including characteristic x state interactions."""
        X = outcome_features(people)
        main = X.to_numpy() @ np.array([self.beta[c] for c in X.columns]) + u
        D = np.array([STATE_INTERACTIONS[c] for c in X.columns])          # features x states
        return main[:, None] + self.interaction_scale * (X.to_numpy() @ D)

    def probability_all_states(self, people, u=None, C=None, R=None):
        """n x 16 matrix: true employment probability of every person in every state."""
        u = np.zeros(len(people)) if u is None else u
        C = self.C0 if C is None else C
        R = np.zeros_like(self.C0) if R is None else R
        score = (self.alpha + self._individual(people, u) + self.gamma[None, :]
                 + self._community(people, C, R))
        return expit(score)

    def probability(self, people, states, u=None, C=None, R=None):
        return self.probability_all_states(people, u, C, R)[np.arange(len(people)), states]

    def _calibrate(self, targets, rng, n):
        """Set alpha and gamma so that, under random quota assignment, the overall employment
        rate and the relative state levels match TARGETS. Every calibration person is
        evaluated in every state, so small states are as precise as large ones."""
        people = sample_attributes(n, rng)
        u = rng.normal(0, self.sigma_u, n)
        q = normalise(STATE_QUOTA)
        base = self._individual(people, u) + self._community(people, self.C0, np.zeros_like(self.C0))
        relative = np.array([targets["state_relative_level"][s] for s in STATES])
        state_target = relative * targets["employment_rate"] / (q @ relative)

        def probs():
            return expit(self.alpha + base + self.gamma[None, :])

        def bisect(error, lo, hi, steps=50):
            for _ in range(steps):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if error(mid) < 0 else (lo, mid)
            return (lo + hi) / 2

        def overall_error(a):
            self.alpha = a
            return (probs() @ q).mean() - targets["employment_rate"]

        for _ in range(8):  # alpha and gamma interact, so iterate until both settle
            self.alpha = bisect(overall_error, -10, 10)
            self.gamma += logit(state_target) - logit(probs().mean(axis=0))
            self.gamma -= q @ self.gamma


def initial_stock():
    """Destatis counts: co-nationals N[state, origin] (16 x 9) and all foreigners T[state]."""
    labels = ORIGINS[:-1] + ["all_foreigners", "Other"]
    counts = pd.DataFrame(DESTATIS_COUNTS, index=labels).T.loc[STATES]
    return counts[ORIGINS].to_numpy(float), counts["all_foreigners"].to_numpy(float)


def update_community(N, T, states, origins, scale):
    """Add this round's placements (each simulated person = `scale` real people).

    Returns the new stock, the new share matrix C and the recent-arrival share matrix R.
    """
    placed = np.zeros_like(N)
    np.add.at(placed, (states, origins), 1)
    N = N + scale * placed
    T = T + scale * placed.sum(axis=1)
    return N, T, N / T[:, None], scale * placed / T[:, None]


def generate(n=5000, seed=42, delta=0.0, assignment="random",
             truth=None, return_details=False):
    """Generate one synthetic cohort.

    n               number of people
    seed            random seed; the same seed gives the same dataset
    delta           community effect (0 = none)
    assignment      "random" or "exact" (see assign_states)
    truth           an existing GroundTruth to reuse; a new one is calibrated otherwise
    return_details  also return the GroundTruth and u_i (useful for simulations)
    """
    rng = np.random.default_rng(seed)
    truth = truth or GroundTruth(delta=delta)

    people = sample_attributes(n, rng)                    # Step 1
    states = assign_states(n, rng, mode=assignment)       # Step 2
    u = rng.normal(0, truth.sigma_u, n)
    p = truth.probability(people, states, u)              # Step 3
    employed = bernoulli(rng, p, n).astype(int)           # Step 4

    people.insert(0, "refugee_id", np.arange(1, n + 1))
    people["assigned_state"] = [STATES[k] for k in states]
    people["true_employment_prob"] = p.round(4)
    people["employed"] = employed
    df = people[COLUMNS]
    return (df, truth, u) if return_details else df


# ---------------------------------------------------------------------------
# Target distributions (for validation)
# ---------------------------------------------------------------------------

def target_distributions():
    """Intended distribution of every attribute, in the same coding as the data."""
    def shares(d):
        return dict(zip(d, normalise(d)))
    return {
        "sex": {"male": P_MALE, "female": 1 - P_MALE},
        "vocational_training": {True: P_VOCATIONAL_TRAINING, False: 1 - P_VOCATIONAL_TRAINING},
        "free_case": {True: P_FREE_CASE, False: 1 - P_FREE_CASE},
        "country_of_origin": shares(COUNTRY_OF_ORIGIN),
        "education_level": shares(EDUCATION_LEVEL),
        "german_speaking": shares(GERMAN_SPEAKING),
        "religion": shares(RELIGION),
        "assigned_state": shares(STATE_QUOTA),
        "age": AGE,
        "school_years": SCHOOL_YEARS,
    }


def summary(df):
    """Print generated vs. target shares."""
    for column, target in target_distributions().items():
        if column in ("age", "school_years"):
            print(f"\n{column}: mean {df[column].mean():.2f} (target {target['mean']})")
            continue
        observed = df[column].value_counts(normalize=True)
        print(f"\n{column}")
        for level, share in target.items():
            print(f"  {str(level):<20} {observed.get(level, 0):6.1%}   target {share:6.1%}")
    education_rank = df["education_level"].map({"low": 0, "medium": 1, "high": 2})
    print(f"\nmean school years by education level: "
          f"{df.groupby('education_level')['school_years'].mean().round(1).to_dict()}")
    print(f"rank correlation education_level / school_years: "
          f"{education_rank.corr(df['school_years'], method='spearman'):.2f}")
    top = df.groupby("country_of_origin")["religion"].agg(lambda r: r.value_counts(normalize=True).idxmax())
    print(f"most common religion by origin: {top.to_dict()}")
    print(f"\nemployment rate: {df['employed'].mean():.1%} "
          f"(mean true probability {df['true_employment_prob'].mean():.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic refugee dataset for Germany.")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--assignment", choices=["random", "exact"], default="random")
    parser.add_argument("--out", default="synthetic_refugees.csv")
    args = parser.parse_args()

    data = generate(args.n, args.seed, args.delta, args.assignment)
    data.to_csv(args.out, index=False)
    print(f"Saved {len(data):,} rows to {args.out}")
    summary(data)
