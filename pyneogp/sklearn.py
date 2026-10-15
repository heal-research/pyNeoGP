"""
A scikit-learn interface to NeoGP: symbolic regression with NSGA2.

    from pyneogp.sklearn import NeoGPRegressor

    model = NeoGPRegressor(generations=50, popsize=500, maxlen=30, seed=1)
    model.fit(X, y)
    model.predict(X)
    model.score(X, y)              # R²
    model.get_best_expression()    # the best expression as printed by NeoGP
    model.get_best_sympy()         # ... as a sympy expression
    model.get_pareto_front()       # all expressions of the Pareto front

NSGA2 evolves a Pareto front of expressions (by default negative log-likelihood vs.
length); the expression of the front with the lowest model-selection criterion (by
default the description length with the rotated parameter complexity) is the model.

Requires scikit-learn and sympy (``pip install "pyneogp[sklearn]"``).
"""

import re

import numpy as np
import sympy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted, validate_data

from pyneogp import neogp
from juliacall import Main as _jl

ROXY_LIKELIHOODS = ("roxy_unif", "roxy_mnr_diagonal")
LIKELIHOODS = ("gaussian", "laplace") + ROXY_LIKELIHOODS

# The parameter complexities of the description length.
_PARAM_COMPLEXITY = {"rot": "param_complexity_rot", "diag": "param_complexity",
                     "rot_scaled": "param_complexity_rot_scaled", "det": "param_complexity_det"}

_print_program_jl = _jl.seval(
    "(indiv, varnames) -> sprint(NeoGP.print_program, NeoGP.code(NeoGP.getprogram(indiv)), varnames)")
_quietly = _jl.seval("f -> redirect_stdout(() -> f(), devnull)")

# The functions of NeoGP's expressions in sympy (`logabs`, `sqrtabs` and `inv` are printed
# as `log(abs(.))`, `sqrt(abs(.))` and `1.0/(.)`, `powabs` as `abs(.) ^ .`).
_SYMPY_FUNCTIONS = {
    "abs": sympy.Abs, "sign": sympy.sign, "neg": lambda a: -a, "sqr": lambda a: a**2,
    "sqrt": sympy.sqrt, "exp": sympy.exp, "log": sympy.log, "log10": lambda a: sympy.log(a, 10),
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan, "asin": sympy.asin,
    "sinh": sympy.sinh, "cosh": sympy.cosh, "tanh": sympy.tanh,
    "NaN": sympy.nan, "Inf": sympy.oo,
}
# A number as printed by Julia, e.g. `-1.5`, `2.0e-5` or `1.0f-5` (Float32); not part of a name.
_NUMBER = re.compile(r"(?<![\w.])(-?)(\d+(?:\.\d*)?(?:[eEf][+-]?\d+)?)")


def _print_program(indiv, varnames):
    """The expression of the individual `indiv` as printed by NeoGP (without the likelihood)."""
    return str(_print_program_jl(indiv, neogp.to_julia(list(varnames))))


def _to_sympy(expr, symbols):
    """Parse the expression `expr` printed by NeoGP with the variables `x0, x1, ...`
    into a sympy expression of the sympy `symbols` of the variables."""
    def number(m):
        n = m.group(2).replace("f", "e")
        # negative numbers get parentheses, because `**` binds stronger than the sign
        return f"({m.group(1)}{n})" if m.group(1) else n

    expr = _NUMBER.sub(number, expr).replace("^", "**")
    names = {f"x{i}": s for i, s in enumerate(symbols)}
    return sympy.parse_expr(expr, local_dict={**_SYMPY_FUNCTIONS, **names})


class NeoGPRegressor(RegressorMixin, BaseEstimator):
    """
    Symbolic regression with NeoGP's NSGA2 as a scikit-learn regressor.

    Parameters
    ----------
    likelihood : str, default="gaussian"
        The likelihood of the data: ``"gaussian"`` or ``"laplace"`` (the noise is estimated
        unless the errors ``yerr`` of the target are given to :meth:`fit`), or the ROXY
        likelihoods ``"roxy_unif"`` (one input) and ``"roxy_mnr_diagonal"``, which require the
        errors ``Xerr`` of the inputs and ``yerr`` of the target.
    intrinsic_scatter : bool, default=False
        Add an estimated intrinsic scatter to the given errors ``yerr`` of the target.
    generations : int, default=100
        The number of generations of NSGA2.
    popsize : int, default=1000
        The population size.
    maxlen : int, default=50
        The maximum length of an expression.
    tournamentsize : int, default=2
        The size of the tournaments of the parent selection.
    mutation_prob : float, default=0.05
        The per-node mutation probability.
    crossover_prob : float, default=0.9
        The probability that an offspring is created by crossover instead of mutation.
    maxeval : int or None, default=None
        The limit of the evaluations of each parameter fit (None: NeoGP's default).
    functionset : str or None, default=None
        The operators as a comma-separated list of NeoGP's opcodes, e.g.
        ``"ADD,SUB,MUL,DIV,EXP,LOG"`` (None: NeoGP's ``DEFAULT_FUNCTIONSET``).
    objective : str, default="nll"
        The two objectives of NSGA2, as the ``--objective`` option of NeoGP's command line:
        ``"nll"`` (also ``"aic"``, ``"bic"``: negative log-likelihood and length), ``"bic_sr"``,
        ``"dl"`` or ``"fbf"`` (the rounded negative log-likelihood and the complexity penalty
        of the criterion).
    model_selection : str, default="dl"
        The criterion that selects the model from the Pareto front: ``"dl"`` (description
        length), ``"nll"``, ``"aic"``, ``"bic"``, ``"bic_sr"`` or ``"fbf"``.
    param_complexity : str, default="rot"
        The parameter complexity of the description length: ``"rot"`` (rotated),
        ``"diag"``, ``"rot_scaled"`` or ``"det"``.
    seed : int, default=-1
        The seed of the random number generator (-1: no seeding).
    threads : int, default=0
        The number of Julia threads of the evaluation (0: all).
    verbose : bool, default=False
        Print the settings of NSGA2 and the best expression of each generation.

    Attributes
    ----------
    nsga2_ : juliacall.AnyValue
        The state of the NSGA2 algorithm (the NeoGP ``NSGA2`` object) after the run.
    n_features_in_ : int
        The number of inputs.
    feature_names_in_ : ndarray of str
        The names of the inputs (only if ``X`` has feature names, e.g. a pandas DataFrame).
    """

    def __init__(self, likelihood="gaussian", intrinsic_scatter=False, generations=100,
                 popsize=1000, maxlen=50, tournamentsize=2, mutation_prob=0.05,
                 crossover_prob=0.9, maxeval=None, functionset=None, objective="nll",
                 model_selection="dl", param_complexity="rot", seed=-1, threads=0,
                 verbose=False):
        self.likelihood = likelihood
        self.intrinsic_scatter = intrinsic_scatter
        self.generations = generations
        self.popsize = popsize
        self.maxlen = maxlen
        self.tournamentsize = tournamentsize
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.maxeval = maxeval
        self.functionset = functionset
        self.objective = objective
        self.model_selection = model_selection
        self.param_complexity = param_complexity
        self.seed = seed
        self.threads = threads
        self.verbose = verbose

    def fit(self, X, y, *, Xerr=None, yerr=None):
        """Run NSGA2 on the inputs `X` (n_samples, n_features) and the target `y`, with the
        standard deviations `Xerr` of the inputs (same shape as `X`, only for the ROXY
        likelihoods, which require it) and `yerr` of the target (n_samples,).  Scalars are
        broadcast to all samples."""
        if self.likelihood not in LIKELIHOODS:
            raise ValueError(f"likelihood must be one of {LIKELIHOODS}, got {self.likelihood!r}")
        if self.likelihood in ROXY_LIKELIHOODS:
            if Xerr is None or yerr is None:
                raise ValueError(f"the {self.likelihood} likelihood requires Xerr and yerr")
        elif Xerr is not None:
            raise ValueError(f"Xerr is not allowed for the {self.likelihood} likelihood")
        if self.generations < 1:
            raise ValueError(f"generations must be >= 1, got {self.generations}")
        objectives = self._objectives()
        criterion = self._criterion()  # checks model_selection and param_complexity

        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)
        n, d = X.shape
        if Xerr is not None:
            Xerr = np.asarray(Xerr, dtype=np.float64)
            if Xerr.ndim == 1 and d == 1:
                Xerr = Xerr.reshape(-1, 1)
            Xerr = np.ascontiguousarray(np.broadcast_to(Xerr, (n, d)))
        if yerr is not None:
            yerr = np.asarray(yerr, dtype=np.float64)
            yerr = np.ascontiguousarray(np.broadcast_to(yerr, (n,)))

        dataset = neogp.Dataset(X, y, Xerr, yerr, self._varnames())
        lik = neogp.create_likelihood(self.likelihood, dataset,
                                      intrinsic_scatter=self.intrinsic_scatter)
        kwargs = dict(seed=self.seed, popsize=self.popsize, maxlen=self.maxlen,
                      tournamentsize=self.tournamentsize, mutation_prob=self.mutation_prob,
                      crossover_prob=self.crossover_prob, maxeval=self.maxeval,
                      threads=self.threads, indiv_loss_func=objectives)
        if self.functionset is not None:
            kwargs["functionset"] = [getattr(neogp, name.strip())
                                     for name in self.functionset.split(",")]

        def create():
            return neogp.NSGA2(lik, d, **kwargs)

        gp = create() if self.verbose else _quietly(create)
        neogp.initialize_b(gp)
        if self.verbose:
            names = self._varnames()

            def log():
                best = min((i for i, _ in neogp.best_individuals(gp)), key=criterion)
                print(f"{gp.fevals} fevals, {self.model_selection} = {criterion(best):.6g}: "
                      f"{_print_program(best, names)}", flush=True)

            neogp.evolve_b(gp, self.generations, iter_callback=log)
        else:
            neogp.evolve_b(gp, self.generations)
        self.nsga2_ = gp
        return self

    def predict(self, X):
        """The predictions of the best expression of the Pareto front for the inputs `X`."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return np.asarray(neogp.predict(self._best(), X))

    def score(self, X, y):
        """The coefficient of determination R² of the predictions for `X`."""
        return r2_score(y, self.predict(X))

    def get_best_expression(self):
        """The best expression of the Pareto front as printed by NeoGP."""
        check_is_fitted(self)
        return _print_program(self._best(), self._varnames())

    def get_best_sympy(self):
        """The best expression of the Pareto front as a sympy expression, with a symbol for
        each input (named as the inputs)."""
        check_is_fitted(self)
        return _to_sympy(self._expression(self._best()), self._symbols())

    def get_pareto_front(self):
        """The expressions of the Pareto front, sorted by the first objective, as a list of
        dicts with the keys ``"expression"`` (as printed by NeoGP), ``"sympy"``,
        ``"objectives"`` (the tuple of the two objectives of NSGA2) and ``"criterion"`` (the
        model-selection criterion, the lowest is the best expression)."""
        check_is_fitted(self)
        criterion, names, symbols = self._criterion(), self._varnames(), self._symbols()
        return [dict(expression=_print_program(indiv, names),
                     sympy=_to_sympy(self._expression(indiv), symbols),
                     objectives=tuple(float(o) for o in objectives),
                     criterion=float(criterion(indiv)))
                for indiv, objectives in neogp.best_individuals(self.nsga2_)]

    def _objectives(self):
        """The objectives of NSGA2, as in NeoGP's command line interface."""
        if self.objective in ("nll", "aic", "bic"):
            return (neogp.negloglik, neogp.individual_length)
        penalties = dict(bic_sr=neogp.BIC_funccompl_penalty,
                         fbf=neogp.fractional_bayes_factor_complexity_penalty)
        if self.objective == "dl":
            penalty = neogp.DescriptionLengthComplexityObjective(self._param_complexity())
        elif self.objective in penalties:
            penalty = penalties[self.objective]
        else:
            raise ValueError("objective must be one of nll, aic, bic, bic_sr, dl, fbf, "
                             f"got {self.objective!r}")
        return (neogp.RoundedObjective(neogp.negloglik, 1), neogp.RoundedObjective(penalty, 1))

    def _criterion(self):
        """The model-selection criterion, a Julia function of an individual."""
        if self.model_selection == "dl":
            return neogp.DescriptionLengthObjective(self._param_complexity())
        criteria = dict(nll=neogp.negloglik, aic=neogp.AIC, bic=neogp.BIC,
                        bic_sr=neogp.BIC_funccompl, fbf=neogp.fractional_bayes_factor)
        if self.model_selection not in criteria:
            raise ValueError("model_selection must be one of dl, nll, aic, bic, bic_sr, fbf, "
                             f"got {self.model_selection!r}")
        return criteria[self.model_selection]

    def _param_complexity(self):
        if self.param_complexity not in _PARAM_COMPLEXITY:
            raise ValueError(f"param_complexity must be one of {tuple(_PARAM_COMPLEXITY)}, "
                             f"got {self.param_complexity!r}")
        return getattr(neogp, _PARAM_COMPLEXITY[self.param_complexity])

    def _best(self):
        """The individual of the Pareto front with the lowest model-selection criterion."""
        criterion = self._criterion()
        return min((indiv for indiv, _ in neogp.best_individuals(self.nsga2_)), key=criterion)

    def _varnames(self):
        names = getattr(self, "feature_names_in_", None)
        if names is None:
            return [f"x{i}" for i in range(self.n_features_in_)]
        return [str(name) for name in names]

    def _symbols(self):
        return [sympy.Symbol(name) for name in self._varnames()]

    def _expression(self, indiv):
        """The expression of `indiv` as printed by NeoGP, with the variables `x0, x1, ...`."""
        return _print_program(indiv, [f"x{i}" for i in range(self.n_features_in_)])
