"""
Symbolic regression of the radial acceleration relation (RAR) of galaxies: the
observed acceleration `gobs` as a function of the baryonic acceleration `gbar`,
both with uncertainties (`e_gbar`, `e_gobs`).  The likelihood marginalizes the
uncertainty of the input (`roxy_mnr_diagonal`) and estimates an intrinsic scatter; the
data are fitted on a logarithmic scale while the models stay on the original scale.

    python examples/RAR.py [extra options of runNeoGP.jl]

The output of the run is written to examples/results/RAR.csv, and the model of the
last generation is plotted over the data (with their errors) into
examples/results/RAR.png.

Environment: JULIA_NUM_THREADS (default: 4), NEOGP_RESULTS_DIR (default: examples/results).
"""

import common
import numpy as np

data = common.data_dir / "RAR.csv"
results = common.results_dir()

# The main command
common.run_neogp([
    data, "gobs",
    "--inputs", "gbar", "--likelihood", "roxy_mnr_diagonal", "--sigma", "e_gbar,e_gobs",
    "--logscale", "xy", "--intrinsic-scatter",
    "--objective", "nll", "--model-selection", "dl", "--generations", 30, "--popsize", 300,
    "--maxlen", 20, "--seed", 1,
], results / "RAR.csv")

###############
# ---- plot the model of the last generation ----
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

f = common.last_model_function(results / "RAR.csv", ["gbar"])
model = common.last_model_title(results / "RAR.csv")
d = common.read_csv(data)  # the columns are gbar, gobs, e_gbar, e_gobs

fig, ax = plt.subplots(figsize=(9, 5.6), dpi=100)
ax.errorbar(d["gbar"], d["gobs"], xerr=d["e_gbar"], yerr=d["e_gobs"], fmt="o", ms=1.5,
            color="gray", elinewidth=0.5, label="data")
x = np.geomspace(0.001, 1000, 500)
ax.plot(x, f(x), "-", color="red", lw=2, label="model", zorder=3)  # over the data
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.001, 1000)
ax.set_xlabel("gbar")
ax.set_ylabel("gobs")
ax.legend(loc="upper left")
ax.set_title("RAR: model of the last generation and data with errors\n" + model, fontsize=8)
fig.tight_layout()
fig.savefig(results / "RAR.png")
print(f"Plot: {results / 'RAR.png'}")
