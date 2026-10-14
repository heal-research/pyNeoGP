"""
Symbolic regression of the friction factor of turbulent flow in rough pipes
(Nikuradse data): the target is a function of the relative roughness `r_k` and
the logarithm of the Reynolds number `log_Re`.  Gaussian noise of unknown variance.

    python examples/nikuradse_1.py

The output of the run is written to examples/results/nikuradse_1.csv, and the
model with the shortest description length of all generations is plotted over the data into
examples/results/nikuradse_1.png (one curve with its 95% prediction interval for each
value of `r_k`).

Environment: JULIA_NUM_THREADS (default: 4), NEOGP_RESULTS_DIR (default: examples/results).
"""

import common
import numpy as np

train = common.data_dir / "nikuradse_1_train.csv"
test = common.data_dir / "nikuradse_1_test.csv"
results = common.results_dir()

best = common.run_nsga2(
    train, "target", results / "nikuradse_1.csv", generations=500, popsize=1000, maxlen=30,
    tournamentsize=2, maxeval=50, seed=1)

#################################################
# ---- plot the best model ----
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

f = common.model_function(best, ["r_k", "log_Re"])
model = common.model_title(best, ["r_k", "log_Re"])
train_data, test_data = common.read_csv(train), common.read_csv(test)
# the values of r_k in the data
rks = np.unique(np.concatenate([train_data["r_k"], test_data["r_k"]]))

fig, ax = plt.subplots(figsize=(9, 5.6), dpi=100)
x = np.linspace(3, 7, 300)
for i, rk in enumerate(rks):
    color = f"C{i}"
    tr = train_data["r_k"] == rk
    te = test_data["r_k"] == rk
    ax.plot(train_data["log_Re"][tr], train_data["target"][tr], "o", ms=3, color=color,
            label=f"{rk:g}")
    ax.plot(test_data["log_Re"][te], test_data["target"][te], "o", ms=4, mfc="none",
            color=color)
    _, low, high = common.prediction_band(best, np.column_stack([np.full_like(x, rk), x]))
    ax.fill_between(x, low, high, color=color, alpha=0.2, lw=0)
    ax.plot(x, f(rk, x), "-", lw=1.5, color=color)
ax.set_xlim(3, 7)
t = np.concatenate([train_data["target"], test_data["target"]])
ax.set_ylim(t.min() - 0.1 * np.ptp(t), t.max() + 0.1 * np.ptp(t))
ax.set_xlabel("log_Re")
ax.set_ylabel("target")
ax.legend(title="r_k", loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_title("Nikuradse: best model (lines), training (filled) and test data\n"
             + model, fontsize=8)
fig.tight_layout()
fig.savefig(results / "nikuradse_1.png")
print(f"Plot: {results / 'nikuradse_1.png'}")
