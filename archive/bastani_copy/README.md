# Replication of Bastani, Bastani & McLaughlin (2026)

Earlier code that reproduces the winner's-curse result of Bastani et al. in their US setting and in a first German version, plus a first multi-round test. Kept for reference; the current code is in `src/refsim/`.

Results (true gain = 0 by design):

| Setting | GBM promises | LASSO promises | Bastani et al. |
|---|---|---|---|
| US | +52% | +11% | +59% / +13% |
| Germany (16 states) | +55% | +14% | – |

Run from this folder: `python run_static.py --setting US --model gbm --reps 50`.
