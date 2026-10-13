"""
Symbolic regression of the one-dimensional Salustowicz function
    f(x) = exp(-x) x^3 cos(x) sin(x) (cos(x) sin(x)^2 - 1)
as used by Vladislavleva et al. (2009): training data on x = 0.05:0.1:10, with
Gaussian noise of 10% of the variance of f added.  The model is plotted on the range
of their test data, x = -0.5:0.05:10.5 (extrapolation at both ends).  The function set
has no power, log or square root, which are undefined for negative inputs.

    python examples/salustowicz.py [extra options of runNeoGP.jl]

The data are generated into examples/data/salustowicz_train.csv.  The output of the
run is written to examples/results/salustowicz.csv, and the model of the last
generation is plotted over the data into examples/results/salustowicz.png.

Environment: JULIA_NUM_THREADS (default: 4), NEOGP_RESULTS_DIR (default: examples/results).
"""

import common
import numpy as np

train = common.data_dir / "salustowicz_train.csv"
results = common.results_dir()


def salustowicz(x):
    return np.exp(-x) * x**3 * np.cos(x) * np.sin(x) * (np.cos(x) * np.sin(x)**2 - 1)


# Generate the training data with noise of 10% of the variance of f
x = 0.05 + 0.1 * np.arange(100)  # 0.05:0.1:10
y = salustowicz(x)
y += np.sqrt(0.1 * np.var(y)) * np.random.default_rng(1).normal(size=len(y))
np.savetxt(train, np.column_stack([x, y]), fmt="%.15g", delimiter=",", header="x,target",
           comments="")

# The main command
common.run_neogp([
    train, "target",
    "--algorithm", "NSGA2", "--functionset", "ADD,SUB,MUL,DIV,EXP,SIN,COS,SQR",
    "--generations", 100, "--popsize", 1000, "--maxlen", 30, "--seed", 1,
], results / "salustowicz.csv")

#################################################
# ---- plot the model of the last generation ----
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

f = common.last_model_function(results / "salustowicz.csv", ["x"])
model = common.last_model_title(results / "salustowicz.csv")
d = common.read_csv(train)

fig, ax = plt.subplots(figsize=(9, 5.6), dpi=100)
x = np.linspace(-0.5, 10.5, 500)
ax.plot(x, salustowicz(x), "-", color="gray", lw=3, label="f(x)")
ax.plot(d["x"], d["target"], "o", ms=3, color="black", label="training data")
ax.plot(x, f(x), "-", color="red", lw=1.5, label="model")
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-1, 1)
ax.set_xlabel("x")
ax.set_ylabel("target")
ax.legend(loc="lower right")
ax.set_title("Salustowicz: model of the last generation and noisy training data\n" + model,
             fontsize=8)
fig.tight_layout()
fig.savefig(results / "salustowicz.png")
print(f"Plot: {results / 'salustowicz.png'}")
