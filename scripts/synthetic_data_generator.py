"""
Synthetic refugee data generator for Germany.

1. individual attributes X_i      (Strasser Ceballos & Kern 2025, Table 5)
2. state assignment T_i           (Königsteiner Schlüssel, BAMF)
3. true employment probability    p_i = sigmoid(alpha + beta'X_i + gamma_T + delta*C_i + u_i)
4. realised employment            Y_i - Bernoulli(p_i)

Beyond the marginals, two dependencies are added without changing any marginal:
education level and years of school are linked through a Gaussian copula, and religion
is drawn conditionally on country of origin (iterative proportional fitting).

Outcome parameters come from published IAB / BA figures where available; the remaining
ones are placeholders and are labelled as such next to their values.

Usage:
    python synthetic_data_generator.py --n 5000 --seed 42 --out synthetic_refugees.csv

    from synthetic_data_generator import generate
    df = generate(n=5000, seed=42)

"""

import argparse

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm, truncnorm


# ---------------------------------------------------------------------------
# Marginal distributions (Strasser Ceballos & Kern 2025, Table 5)
# ---------------------------------------------------------------------------

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
    "Shiite": 60.6, "Sunnite": 7.6, "Other Islamic": 6.7, "Other Christian": 6.7,
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
# Outcome model parameters
# ---------------------------------------------------------------------------

# Coefficients on the log-odds scale.
#
# Education and German language: average marginal effects on the probability of employment from the IAB-BAMF-SOEP Survey of Refugees (IAB-Forum, "10 Jahre Fluchtmigration 2015", 2025):
#   medium school education +3 pp, higher school education +5 pp, university +6 pp, good or very good German +7 pp.
# Converted to log-odds with beta = AME / (p * (1 - p)), using p = 0.4 as the approximate mean employment rate in that sample. Our "high" level combines higher school and tertiary education,
# so it takes the midpoint (+5.5 pp). The source compares good/very good German with all lower
# levels, so "poorly" and "fairly" stay at the reference.
# "male" is calibrated to the gender gap (see TARGETS). Age and vocational training have no published estimate in this form yet and are placeholders. School years carry no separate
# effect, because they are correlated with education_level (see EDUCATION_SCHOOL_CORRELATION).

AME_TO_LOGIT = 1 / (0.4 * 0.6)
BETA = {
    "male": None,
    "age_per_10_years_over_30": -0.20,
    "education_medium": 0.030 * AME_TO_LOGIT,
    "education_high": 0.055 * AME_TO_LOGIT,
    "school_years_centered": 0.0,
    "vocational_training": 0.12,
    "german_poorly": 0.0,
    "german_fairly": 0.0,
    "german_good": 0.070 * AME_TO_LOGIT,
    "free_case": 0.0,
}
# Religion is part of the dataset but not of the outcome equation.

SIGMA_U = 0.5  # standard deviation of unobserved individual differences u_i (placeholder)

# Employment rate of nationals of the eight main asylum countries by state, January 2026
# (Bundesagentur für Arbeit, "Beschäftigte aus den 8 nichteuropäischen Asylherkunftsländern",
# latest month with final values). Only the relative differences between states are used.
STATE_EMPLOYMENT_RATE_BA = {
    "NW": 49.2, "BY": 55.3, "BW": 52.7, "NI": 48.0, "HE": 48.4, "BE": 49.9, "SN": 46.4, "RP": 49.6,
    "SH": 48.5, "BB": 44.2, "ST": 44.1, "HH": 54.8, "TH": 50.1, "MV": 44.6, "SL": 45.1, "HB": 46.3,
}

# Calibration targets for alpha, beta_male and gamma.
TARGETS = {
    "employment_rate": 0.15,        # baseline in the Strasser Ceballos & Kern case study
    "women_to_men_ratio": 31 / 75,  # IAB: 31% of women vs 75% of men employed (2015 cohort, 2022)
    "state_relative_level": {state: rate / max(STATE_EMPLOYMENT_RATE_BA.values())
                             for state, rate in STATE_EMPLOYMENT_RATE_BA.items()},
}

# Dependence between attributes. Table 5 only gives marginal distributions, so these are modelling choices that keep every marginal unchanged.

# Education level and years of school are linked through a Gaussian copula with this correlation (placeholder, to be replaced with an estimate from SOEP).
EDUCATION_SCHOOL_CORRELATION = 0.6

# Religion depends on country of origin. The rows below give a rough plausible shape of religion within each origin (placeholder weights, not data). Iterative proportional fitting
# then rescales them so that both the origin and the religion marginals match Table 5 exactly. Columns follow the order of RELIGION.

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Shape parameters of a truncated normal on [low, high] whose mean after truncation is `mean`.

    Truncation shifts the mean (e.g. N(32, 10) cut at 18 has mean ~33.6), so the location
    parameter is adjusted until the truncated mean matches the target.
    """
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


# ---------------------------------------------------------------------------
# individual attributes
# ---------------------------------------------------------------------------

def sample_attributes(n, rng):
    people = pd.DataFrame({
        "sex": np.where(bernoulli(rng, P_MALE, n), "male", "female"),
        "age": truncated_normal(rng, n, **AGE).round(1),
        "country_of_origin": inverse_transform_sample(rng, COUNTRY_OF_ORIGIN, n),
        "vocational_training": bernoulli(rng, P_VOCATIONAL_TRAINING, n),
        "german_speaking": inverse_transform_sample(rng, GERMAN_SPEAKING, n),
        "free_case": bernoulli(rng, P_FREE_CASE, n),
    })

    # Education level and years of school: Gaussian copula. Two correlated standard normals are turned into uniforms, and each uniform is mapped through its own marginal distribution.
    rho = EDUCATION_SCHOOL_CORRELATION
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    u = norm.cdf(z)
    people["education_level"] = categories_from_uniform(u[:, 0], EDUCATION_LEVEL)
    people["school_years"] = truncated_normal_from_uniform(u[:, 1], **SCHOOL_YEARS).round(1)

    # Religion drawn conditionally on country of origin.
    conditional = religion_given_origin()
    religion = np.empty(n, dtype=object)
    for i, origin in enumerate(ORIGINS):
        members = (people["country_of_origin"] == origin).to_numpy()
        religion[members] = inverse_transform_sample(
            rng, dict(zip(RELIGION, conditional[i])), members.sum())
    people["religion"] = religion

    return people[["sex", "age", "country_of_origin", "education_level", "school_years",
                   "vocational_training", "german_speaking", "free_case", "religion"]]


# ---------------------------------------------------------------------------
# state assignment
# ---------------------------------------------------------------------------

def capacities(n):
    """c_k = floor(N * q_k); leftover seats go to the largest remainders so that sum(c_k) = N."""
    q = normalise(STATE_QUOTA)
    c = np.floor(n * q).astype(int)
    c[np.argsort(-(n * q - c))[: n - c.sum()]] += 1
    return c


def assign_states(n, rng, mode="random"):
    """Quasi-random allocation by quota.

    "random": each person is drawn independently with probabilities q.
    "exact":  state sizes equal the capacities c_k; people are shuffled into the seats.
    """
    if mode == "random":
        return inverse_transform_sample(rng, dict(zip(range(len(STATES)), STATE_QUOTA.values())), n).astype(int)
    if mode == "exact":
        return rng.permutation(np.repeat(np.arange(len(STATES)), capacities(n)))
    raise ValueError("mode must be 'random' or 'exact'")


# ---------------------------------------------------------------------------
# ground-truth outcome engine
# ---------------------------------------------------------------------------

def outcome_features(people):
    return pd.DataFrame({
        "male": (people["sex"] == "male").astype(float),
        "age_per_10_years_over_30": np.maximum(people["age"] - 30, 0) / 10,
        "education_medium": (people["education_level"] == "medium").astype(float),
        "education_high": (people["education_level"] == "high").astype(float),
        "school_years_centered": people["school_years"] - 10,
        "vocational_training": people["vocational_training"].astype(float),
        "german_poorly": (people["german_speaking"] == "poorly").astype(float),
        "german_fairly": (people["german_speaking"] == "fairly").astype(float),
        "german_good": (people["german_speaking"] == "good").astype(float),
        "free_case": people["free_case"].astype(float),
    })


class GroundTruth:
    """p_i = sigmoid(alpha + beta'X_i + gamma[T_i] + delta*C_i + delta2*C_i^2 + u_i)

    delta > 0 means community support, delta < 0 labour-market congestion.
    delta2 adds an optional quadratic term (delta > 0, delta2 < 0: small communities
    help, large ones hurt). alpha, beta["male"] and gamma are calibrated to TARGETS.
    """

    def __init__(self, delta=0.0, delta2=0.0, beta=BETA, sigma_u=SIGMA_U, targets=TARGETS,
                 calibration_seed=12345, calibration_size=50_000):
        self.beta = dict(beta)
        self.delta = delta
        self.delta2 = delta2
        self.sigma_u = sigma_u
        self.C0 = initial_community_matrix()
        self.alpha = 0.0
        self.gamma = np.zeros(len(STATES))
        self._calibrate(targets, np.random.default_rng(calibration_seed), calibration_size)

    def probability(self, people, states, u=None, C=None):
        """True employment probability of each person in the given state."""
        C = self.C0 if C is None else C
        u = np.zeros(len(people)) if u is None else u
        X = outcome_features(people)
        beta = np.array([self.beta[name] for name in X.columns])
        origin = people["country_of_origin"].map({o: i for i, o in enumerate(ORIGINS)}).to_numpy()
        community = C[states, origin]
        score = (self.alpha + X.to_numpy() @ beta + self.gamma[states]
                 + self.delta * community + self.delta2 * community ** 2 + u)
        return expit(score)

    def probability_all_states(self, people, u=None, C=None):
        """n x 16 matrix of potential outcomes: the true probability in every state."""
        n = len(people)
        return np.column_stack([self.probability(people, np.full(n, k), u, C)
                                for k in range(len(STATES))])

    def _calibrate(self, targets, rng, n):
        """Set alpha, beta["male"] and gamma so that, under random quota assignment, the
        overall rate, the women/men ratio and the relative state levels match TARGETS.

        Every calibration person is evaluated in every state, so small states are as
        precise as large ones.
        """
        people = sample_attributes(n, rng)
        u = rng.normal(0, self.sigma_u, n)
        q = normalise(STATE_QUOTA)
        male = (people["sex"] == "male").to_numpy()

        X = outcome_features(people)
        other = [name for name in X.columns if name != "male"]
        base = X[other].to_numpy() @ np.array([self.beta[name] for name in other]) + u
        origin = people["country_of_origin"].map({o: i for i, o in enumerate(ORIGINS)}).to_numpy()
        community = self.C0[:, origin].T                      # n x 16
        base = base[:, None] + self.delta * community + self.delta2 * community ** 2

        relative = np.array([targets["state_relative_level"][s] for s in STATES])
        state_target = relative * targets["employment_rate"] / (q @ relative)

        def probabilities():
            return expit(self.alpha + base + self.beta["male"] * male[:, None] + self.gamma[None, :])

        def bisect(error, low, high, steps=50):
            for _ in range(steps):
                mid = (low + high) / 2
                low, high = (mid, high) if error(mid) < 0 else (low, mid)
            return (low + high) / 2

        def overall_error(a):
            self.alpha = a
            return (probabilities() @ q).mean() - targets["employment_rate"]

        def gender_error(b):
            self.beta["male"] = b
            expected = probabilities() @ q
            return targets["women_to_men_ratio"] - expected[~male].mean() / expected[male].mean()

        self.beta["male"] = 0.0
        for _ in range(8):  # the targets interact, so iterate until they settle
            self.alpha = bisect(overall_error, -10, 10)
            self.beta["male"] = bisect(gender_error, 0, 5)
            self.gamma += logit(state_target) - logit(probabilities().mean(axis=0))
            self.gamma -= q @ self.gamma


# ---------------------------------------------------------------------------
# full generator
# ---------------------------------------------------------------------------

def generate(n=5000, seed=42, delta=0.0, delta2=0.0, assignment="random",
             truth=None, return_details=False):
    """Generate one synthetic cohort.

    n               number of people
    seed            random seed; the same seed gives the same dataset
    delta, delta2   community effect (0 = none)
    assignment      "random" or "exact" (see assign_states)
    truth           an existing GroundTruth to reuse; a new one is calibrated otherwise
    return_details  also return the GroundTruth and u_i (useful for simulations)
    """
    rng = np.random.default_rng(seed)
    truth = truth or GroundTruth(delta=delta, delta2=delta2)

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
    parser.add_argument("--delta2", type=float, default=0.0)
    parser.add_argument("--assignment", choices=["random", "exact"], default="random")
    parser.add_argument("--out", default="synthetic_refugees.csv")
    args = parser.parse_args()

    data = generate(args.n, args.seed, args.delta, args.delta2, args.assignment)
    data.to_csv(args.out, index=False)
    print(f"Saved {len(data):,} rows to {args.out}")
    summary(data)
