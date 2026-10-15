import os

import numpy as np
import pytest

sympy = pytest.importorskip("sympy")
pytest.importorskip("sklearn")

from sklearn.base import clone  # noqa: E402

from pyneogp.sklearn import NeoGPRegressor, _to_sympy  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "data")


@pytest.fixture
def product_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 5, size=(100, 2))
    y = X[:, 0] * X[:, 1] + 0.01 * rng.normal(size=100)
    return X, y


def test_fit_predict_score(product_data):
    X, y = product_data
    model = NeoGPRegressor(generations=10, popsize=100, maxlen=15, seed=1,
                           functionset="ADD,SUB,MUL")
    assert model.fit(X, y) is model
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert model.score(X, y) > 0.999
    assert "x0" in model.get_best_expression()

    # the sympy expression computes the predictions
    expr = model.get_best_sympy()
    x0, x1 = sympy.symbols("x0 x1")
    assert expr.free_symbols <= {x0, x1}
    f = sympy.lambdify([x0, x1], expr, "numpy")
    np.testing.assert_allclose(np.broadcast_to(f(X[:, 0], X[:, 1]), y.shape), pred, rtol=1e-6)

    front = model.get_pareto_front()
    assert len(front) >= 1
    assert {"expression", "sympy", "objectives", "criterion"} == set(front[0])
    best = min(front, key=lambda entry: entry["criterion"])
    assert best["expression"] == model.get_best_expression()
    assert best["sympy"] == expr
    assert all(len(entry["objectives"]) == 2 for entry in front)

    assert clone(model).get_params() == model.get_params()


def test_feature_names(product_data):
    pd = pytest.importorskip("pandas")
    X, y = product_data
    df = pd.DataFrame(X, columns=["a", "b c"])
    model = NeoGPRegressor(generations=3, popsize=50, maxlen=10, seed=1).fit(df, y)
    assert model.get_best_sympy().free_symbols <= {sympy.Symbol("a"), sympy.Symbol("b c")}


@pytest.mark.parametrize("model_selection", ["nll", "aic", "bic", "bic_sr", "fbf"])
def test_model_selection(product_data, model_selection):
    X, y = product_data
    model = NeoGPRegressor(generations=2, popsize=50, maxlen=10, seed=1, objective="dl",
                           model_selection=model_selection).fit(X, y)
    front = model.get_pareto_front()
    assert min(front, key=lambda e: e["criterion"])["expression"] == model.get_best_expression()


def test_errors(product_data):
    X, y = product_data
    with pytest.raises(ValueError, match="Xerr is not allowed"):
        NeoGPRegressor().fit(X, y, Xerr=np.ones_like(X))
    with pytest.raises(ValueError, match="requires Xerr and yerr"):
        NeoGPRegressor(likelihood="roxy_mnr_diagonal").fit(X, y, yerr=0.1)
    with pytest.raises(ValueError, match="likelihood must be one of"):
        NeoGPRegressor(likelihood="normal").fit(X, y)
    with pytest.raises(ValueError, match="model_selection"):
        NeoGPRegressor(model_selection="xyz").fit(X, y)
    model = NeoGPRegressor(generations=1, popsize=20, maxlen=10, seed=1)
    model.fit(X, y, yerr=0.01)  # Gaussian likelihood with the given error of the target
    with pytest.raises(ValueError):
        model.predict(X[:, :1])


def test_roxy():
    path = os.path.join(DATA, "RAR.csv")
    if not os.path.isfile(path):
        pytest.skip("RAR.csv is only available in a checkout of pyNeoGP")
    gbar, gobs, e_gbar, e_gobs = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
    model = NeoGPRegressor(likelihood="roxy_mnr_diagonal", intrinsic_scatter=True,
                           generations=3, popsize=50, maxlen=10, seed=1)
    model.fit(gbar.reshape(-1, 1), gobs, Xerr=e_gbar, yerr=e_gobs)
    assert np.all(np.isfinite([e["criterion"] for e in model.get_pareto_front()]))
    assert model.predict(gbar.reshape(-1, 1)).shape == gobs.shape


def test_to_sympy():
    a, b = sympy.symbols("a b")
    assert _to_sympy("(-1.5 ^ x0)", [a, b]) == sympy.Float(-1.5) ** a
    assert _to_sympy("abs(x0) ^ -2.5", [a, b]) == sympy.Abs(a) ** sympy.Float(-2.5)
    assert _to_sympy("(log(abs(x1)) * 1.0f-5)", [a, b]) == sympy.Float(1e-5) * sympy.log(sympy.Abs(b))
    assert _to_sympy("(1.0/(sqr(x1)))", [a, b]) == sympy.Float(1.0) / b**2
    assert _to_sympy("((-0.5 * x0) - -3.0e-7)", [a, b]) == sympy.Float(-0.5) * a + sympy.Float(3e-7)
