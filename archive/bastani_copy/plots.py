import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# static histograms (Bastani Fig. 2 analogue)
fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
for ax, (s, m) in zip(axes, [("US", "gbm"), ("US", "lasso"), ("DE", "gbm"), ("DE", "lasso")]):
    d = pd.read_csv(f"results/static_{s}_{m}.csv")
    ax.hist(d.gain_ipw_pct, bins=20, alpha=.5, label="IPW", color="tab:green")
    ax.hist(d.gain_model_based_pct, bins=20, alpha=.7, label="model-based", color="tab:purple")
    ax.axvline(d.gain_true_policy_pct.mean(), color="k", ls="--", label="true")
    ax.set_title(f"{s} · {m.upper()}  (model-based {d.gain_model_based_pct.mean():+.0f}%)")
    ax.set_xlabel("% change in employment")
axes[0].legend(); axes[0].set_ylabel("replications")
plt.tight_layout(); plt.savefig("results/static_null_world.png", dpi=150)

# dynamic
d = pd.read_csv("results/dynamic_DE.csv")
g = d.groupby(["condition", "round"])[["gain_true_policy_pct", "gain_model_promised_pct"]].mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for c, sty in zip(["B_static", "C_endogenous", "D_exogenous"], ["-o", "-s", "-^"]):
    x = g[g.condition == c]
    axes[0].plot(x["round"], x.gain_true_policy_pct, sty, label=c)
    axes[1].plot(x["round"], x.gain_model_promised_pct, sty, label=c)
axes[0].set_title("TRUE gain vs random quota assignment"); axes[1].set_title("Gain PROMISED by the model")
for ax in axes: ax.set_xlabel("round"); ax.set_ylabel("% gain"); ax.axhline(0, color="grey", lw=.5)
axes[0].legend(); plt.tight_layout(); plt.savefig("results/dynamic_DE.png", dpi=150)
print("ok")
