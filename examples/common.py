"""Helpers of the example scripts (imported, not run)."""

import os
import pathlib
import re
import textwrap

import numpy as np
import matplotlib  # noqa: F401  (imported first: fails before the run if it is missing)

# Julia threads of the examples (default: 4); must be set before pyneogp starts Julia.
os.environ.setdefault("JULIA_NUM_THREADS", "4")

from pyneogp import neogp  # noqa: E402  (before juliacall, see pyneogp/__init__.py)
from juliacall import Main as jl  # noqa: E402

root = pathlib.Path(__file__).resolve().parent.parent
data_dir = root / "examples" / "data"


def results_dir():
    """Directory of the output of the examples (the log of the run and the plot)."""
    path = pathlib.Path(os.environ.get("NEOGP_RESULTS_DIR", root / "examples" / "results"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_nsga2(train, target, output, *, inputs=(), likelihood="gaussian", sigma=None,
              logscale="", intrinsic_scatter=False,
              functionset="ADD,SUB,MUL,DIV,EXP,LOG,SQRT,POW", generations=100, popsize=1000,
              tournamentsize=2,
              maxlen=50, maxeval=20, seed=-1, param_complexity=neogp.param_complexity_rot):
    """Run NSGA2 (negative log-likelihood and length as objectives) on the CSV file `train`
    and return the model with the shortest description length of all generations.

    `sigma` is a list of the columns with the errors, `logscale` is `x`, `y` or `xy`.  The best
    model of each generation is written to stdout and to the CSV file `output`."""
    dataset = neogp.load_dataset(jl.Float32, str(train), target, likelihood=likelihood,
                                 sigma=sigma, **({"inputs": list(inputs)} if inputs else {}))
    if logscale:
        scale = dict(x="x" in logscale, y="y" in logscale)
        dataset = neogp.logscale(dataset, **scale)
    lik = neogp.create_likelihood(likelihood, dataset, intrinsic_scatter=intrinsic_scatter)
    if logscale:
        lik = neogp.LogScaledLikelihood(lik, **scale)  # the models stay on the original scale

    gp = neogp.NSGA2(lik, dataset.X.shape[1], seed=seed, popsize=popsize, maxlen=maxlen,
                     indiv_loss_func=(neogp.negloglik, neogp.individual_length),
                     maxeval=maxeval, tournamentsize=tournamentsize,
                     functionset=[getattr(neogp, name) for name in functionset.split(",")],
                     threads=int(jl.Threads.nthreads()))
    description_length = neogp.DescriptionLengthObjective(param_complexity)

    lines = ["gen,fevals,size,DL,expression"]
    best, best_dl, best_gen = None, np.inf, 0

    def log():
        nonlocal best, best_dl, best_gen
        b = min((i for i, _ in neogp.best_individuals(gp)), key=description_length)
        dl = description_length(b)
        if dl < best_dl:
            best, best_dl, best_gen = neogp.copy_indiv(b), dl, len(lines)
        lines.append(f'{len(lines)},{gp.fevals},{neogp.individual_length(b)},{dl},'
                     f'"{neogp.tostring(b, dataset.varnames)}"')
        print(lines[-1], flush=True)

    print(lines[0], flush=True)
    neogp.initialize_b(gp)
    log()
    neogp.evolve_b(gp, generations - 1, iter_callback=log)
    pathlib.Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Best model of generation {best_gen}: DL = {best_dl}")
    return best


def prediction_band(model, X, alpha=0.05):
    """The mean prediction and the limits of the `1 - alpha` prediction interval of `model` (as
    returned by `run_nsga2`) at the rows of `X`, from `SymRegLikelihoods.prediction_intervals`.
    Raises `juliacall.JuliaError` for likelihoods without a constant observation scatter (the
    ROXY likelihoods, errors of the target that differ between observations)."""
    X = np.asarray(X, dtype=np.float32)
    lik = neogp.getlossfunction(model)
    prog = neogp.getprogram(model)
    params = neogp.extractparam(jl.Float32, model)
    # for a log-scaled likelihood the interval is that of the log-scaled model of the wrapped
    # likelihood, on the log-scaled inputs, and is transformed back if the target is log-scaled
    scaled = jl.isa(lik, neogp.LogScaledLikelihood.julia)
    if scaled:
        prog = neogp.evaluation_model(lik, prog)
        if lik.logx:
            X = np.log10(X)
        lik = lik.likelihood
    bands = [np.asarray(v) for v in
             neogp.SL.prediction_intervals(lik, prog, neogp.to_julia(X), params, alpha=alpha)[:3]]
    return [10 ** v for v in bands] if scaled and neogp.getlossfunction(model).logy else bands


def model_expression(model, varnames):
    """The expression of the individual `model` as printed by NeoGP, without the name of the
    likelihood and the likelihood parameters that follow it."""
    return re.sub(r" [^ ]+ (Float[0-9]+)?\[.*\]$", "", str(neogp.tostring(model, varnames)))


def model_title(model, varnames, width=110):
    """The expression of `model` for the title of a plot, wrapped at spaces into lines of at
    most `width` characters."""
    return textwrap.fill(model_expression(model, varnames), width)


# The functions of NeoGP's expressions (`inv` is printed as a division).
_functions = {name: getattr(np, name) for name in
              ("sqrt", "log", "exp", "abs", "sin", "cos", "tan", "sinh", "cosh", "tanh")}
_functions["sqr"] = np.square


def model_function(model, varnames):
    """The individual `model` as a numpy function of the variables `varnames`.  The expression is translated to Python:
      - Julia prints small `Float32` numbers as `1.0f-5`,
      - negative numbers get parentheses, because `**` binds stronger than the sign,
      - `^` is `**` in Python."""
    expr = model_expression(model, varnames)
    expr = re.sub(r"([0-9])f(-?[0-9])", r"\1e\2", expr)
    expr = re.sub(r"(^|[^0-9eE.)A-Za-z_])(-[0-9.]+(e-?[0-9]+)?)", r"\1(\2)", expr)
    expr = expr.replace("^", "**")
    print(f"Model: {expr}")
    code = compile(expr, "<model>", "eval")

    def model(*args):
        with np.errstate(all="ignore"):  # NaN outside of the domain, as in Julia
            value = eval(code, {"__builtins__": {}, **_functions}, dict(zip(varnames, args)))
        return np.broadcast_to(value, np.broadcast(*args).shape)

    return model


def read_csv(path):
    """The columns of the CSV file `path` (with a header) as a dict of numpy arrays."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: data[name] for name in data.dtype.names}
