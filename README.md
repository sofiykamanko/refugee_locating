# Refugee Location Matching Under Endogenous Shift
Can a refugee matching algorithm make its own results worse over time?

Algorithms that match refugees to locations learn from past placements, but their own decisions change the conditions they learn from. This project studies whether such systems can degrade over time through these feedback effects, using Germany as a case.

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```
## Tests

```bash
uv run pytest
```

Runs a smoke simulation (2 rounds, 500 arrivals, 500 in history, seed 0), checks the matching solver against scipy and checks that the experiment grids expand correctly. 

## Running simulations

```bash
uv run python -m refugee_sim.simulation configs/main.yaml [configs/history.yaml ...]
```

Each config is expanded into its grid points, and every point is run for every seed. Each run is saved as `results/<name>/<run_key>.csv` (one row per round: settings, all metrics and the placement counts `place__<state>__<origin>`) together with `<run_key>.yaml`, the exact config it ran with. The `run_key` encodes the main settings plus a hash of the full config, so runs never overwrite each other. Runs whose csv already exists are skipped, so an interrupted experiment can simply be restarted.

From Python:

```python
from refugee_sim.simulation import load_config, run_simulation, load_results

df = run_simulation(load_config('configs/default.yaml'), seed=0)
grid = load_results('results/main')
```

## Configs

Every config starts from `default.yaml` (`base: default.yaml`) and only states what it changes; `grid:` lists the values to combine. Unknown keys raise an error, so typos do not silently fall back to defaults.

| File | Purpose |
|---|---|
| `default.yaml` | Agreed defaults: 10 rounds, 2,000 arrivals per round, 10,000 in history, 5 seeds (raise to 10–20 for final figures), linear community effect with δ = 0, SOEP interactions, fixed model that does not see the community. |
| `smoke.yaml` | Quick check: 2 rounds, 500 arrivals, 500 in history, 1 seed. Used by the tests. |
| `main.yaml` | Main experiment, 45 points: δ ∈ {−0.3, −0.15, 0, 0.15, 0.3} × interactions (SOEP, stylised ×4, none) × retraining (fixed, every 3 rounds, every round). |
| `history.yaml` | Size of the training history (3,000 / 30,000 / 100,000) for SOEP and ×4 interactions, model retrained every round. |
| `forms.yaml` | Alternative community-effect shapes under SOEP interactions: inverted U, saturating (±), stronger for women, established vs recent arrivals; fixed vs retrained every round. |
| `community_model.yaml` | Workstream A, step 1: the policy model sees the community (`policy.features`: base / community / community_recent), 54 points: features × δ ∈ {−0.3, 0, 0.3} × interactions × fixed vs every round. |
| `learning_delta.yaml` | Workstream A, step 5: is δ estimated correctly from the algorithm's own data (`delta_hat`)? Model vs random placement, fixed vs every round, 5 values of δ, 20 points. |
| `forward_looking.yaml` | Workstream A, step 3: greedy vs forward-looking placement (`policy.lookahead`: 0 / 3 / 10 future rounds), learned model vs oracle (`policy.planner`), δ ∈ {−0.3, 0, 0.3}, SOEP, 20 rounds, 18 points. |

## Code

| Module | Content |
|---|---|
| `src/refugee_sim/generator.py` | Synthetic population, calibrated ground-truth employment model, community stock updates |
| `src/refugee_sim/policy.py` | Gradient-boosting policy model (optionally seeing the community), greedy and forward-looking capacity-constrained matching |
| `src/refugee_sim/simulation.py` | Feedback loop, `run_simulation(config, seed)`, config loading, results |

`notebooks/analysis.ipynb` produces the figures from `results/` only (`uv run jupyter lab`); notebooks contain no simulation code. Data sources are described in `data/README.md`.
