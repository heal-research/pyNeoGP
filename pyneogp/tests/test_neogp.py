import os

import numpy as np
import pytest
from juliacall import JuliaError, Main as jl

from pyneogp import neogp

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "data")


def data_file(name):
    path = os.path.join(DATA, name)
    if not os.path.isfile(path):
        pytest.skip(f"{name} is only available in a checkout of pyNeoGP")
    return path


@pytest.fixture
def product_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 5, size=(100, 2))
    y = X[:, 0] * X[:, 1] + 0.01 * rng.normal(size=100)
    return X, y


def test_module():
    assert str(neogp.julia_module) == "NeoGP"
    assert jl.Threads.nthreads() >= 1
    assert "evolve_b" in dir(neogp) and "NSGA2" in dir(neogp)
    assert "Two-objective genetic programming" in neogp.NSGA2.__doc__


def test_to_julia():
    assert jl.isa(neogp.to_julia(np.zeros((3, 2))), jl.Matrix[jl.Float64])
    assert jl.isa(neogp.to_julia(np.zeros(3, dtype=np.float32)), jl.Vector[jl.Float32])
    assert jl.isa(neogp.to_julia(["a", "b"]), jl.Vector[jl.String])
    assert jl.isa(neogp.to_julia([0.1, 0.2]), jl.Vector[jl.Float64])
    assert jl.isa(neogp.to_julia(np.array(["a", "b"])), jl.Vector[jl.String])
    assert jl.isa(neogp.to_julia([neogp.ADD, neogp.MUL]), jl.Vector[jl.typeof(neogp.ADD)])
    f, g = neogp.to_julia((neogp.negloglik, neogp.individual_length))
    assert jl.isa(f, jl.Function) and jl.isa(g, jl.Function)
    assert neogp.to_julia(np.float32(0.5)) == 0.5 and isinstance(neogp.to_julia(np.int64(3)), int)
    # row-major numpy arrays keep their layout
    X = np.arange(6.0).reshape(2, 3)
    assert neogp.to_julia(X)[1, 2] == X[1, 2]


@pytest.mark.parametrize("dtype, jltype", [(np.float64, "Float64"), (np.float32, "Float32")])
def test_dataset_from_numpy(product_data, dtype, jltype):
    X, y = product_data
    ds = neogp.Dataset(X.astype(dtype), y.astype(dtype), None, None, ["a", "b"])
    assert jl.isa(ds, neogp.Dataset[getattr(jl, jltype)])
    assert list(ds.varnames) == ["a", "b"]
    np.testing.assert_allclose(np.asarray(ds.X), X.astype(dtype))


def test_nsga2_numpy(product_data):
    X, y = product_data
    gp = neogp.NSGA2(X, y, popsize=np.int64(100), maxlen=15, seed=1,
                     functionset=[neogp.ADD, neogp.SUB, neogp.MUL],
                     indiv_loss_func=(neogp.negloglik, neogp.individual_length))
    neogp.initialize_b(gp)
    generations = []
    neogp.evolve_b(gp, 10, iter_callback=lambda: generations.append(len(generations)))
    assert len(generations) == 10

    front = neogp.best_individuals(gp)
    assert len(front) >= 1
    indiv, (nll, length) = front[0]
    assert length == neogp.individual_length(indiv)
    pred = np.asarray(neogp.predict(indiv, X))
    assert pred.shape == y.shape
    assert neogp.r2_score(y, pred) > 0.999
    assert "a" in neogp.tostring(indiv, ["a", "b"])


def test_basic_algorithm(product_data):
    X, y = product_data
    likelihood = neogp.create_likelihood("gaussian", neogp.Dataset(X, y, None, None, ["a", "b"]))
    gp = neogp.BasicAlgorithm(likelihood, 2, popsize=np.int64(100), maxlen=15, seed=1,
                              indiv_loss_func=neogp.description_length)
    neogp.initialize_b(gp)
    neogp.evolve_b(gp, 10)
    [(indiv, dl)] = neogp.best_individuals(gp)
    assert dl == pytest.approx(neogp.description_length(indiv))
    assert neogp.r2_score(y, neogp.predict(indiv, X)) > 0.99


def test_reproducible(product_data):
    X, y = product_data

    def run():
        gp = neogp.NSGA2(X, y, popsize=50, maxlen=15, seed=42)
        neogp.initialize_b(gp)
        neogp.evolve_b(gp, 3)
        return [neogp.tostring(indiv) for indiv, _ in neogp.best_individuals(gp)]

    assert run() == run()


def test_quickstart_nikuradse():
    # the Julia example of QUICKSTART.md
    train = neogp.load_dataset(jl.Float32, data_file("nikuradse_1_train.csv"), "target")
    gp = neogp.NSGA2(neogp.create_likelihood("gaussian", train), train.X.shape[1],
                     popsize=100, maxlen=20, seed=1)
    neogp.initialize_b(gp)
    neogp.evolve_b(gp, 5)
    front = neogp.best_individuals(gp)
    best, _ = min(front, key=lambda entry: neogp.description_length(entry[0]))
    expr = neogp.tostring(best, train.varnames)
    assert isinstance(expr, str) and expr
    pred = np.asarray(neogp.predict(best, train.X))
    assert pred.dtype == np.float32 and pred.shape == (len(train.y),)


def test_rar_roxy_logscale():
    # the RAR example of QUICKSTART.md: errors in the inputs, intrinsic scatter, log scale
    data = neogp.load_dataset(jl.Float64, data_file("RAR.csv"), "gobs", inputs=["gbar"],
                              sigma=["e_gbar", "e_gobs"], likelihood="roxy_mnr_diagonal")
    scaled = neogp.logscale(data, x=True, y=True)
    likelihood = neogp.LogScaledLikelihood(
        neogp.create_likelihood("roxy_mnr_diagonal", scaled, intrinsic_scatter=True), x=True, y=True)
    gp = neogp.NSGA2(likelihood, 1, popsize=50, maxlen=10, seed=1)
    neogp.initialize_b(gp)
    neogp.evolve_b(gp, 3)
    indiv, _ = neogp.best_individuals(gp)[0]
    assert np.isfinite(neogp.negloglik(indiv))


def test_errors_are_raised(product_data):
    X, y = product_data
    with pytest.raises(JuliaError, match="popsize must be >= 1"):
        neogp.NSGA2(X, y, popsize=0)
    with pytest.raises(JuliaError, match="DimensionMismatch"):
        neogp.Dataset(X, y[:-1], None, None, ["a", "b"])


def test_cli(tmp_path):
    # Julia writes to its own stdout stream, which pytest does not capture
    output = tmp_path / "out.csv"
    run_redirected = jl.seval("(f, path) -> open(io -> redirect_stdout(() -> f(), io), path, \"w\")")
    run_redirected(lambda: neogp.main([data_file("nikuradse_1_train.csv"), "target",
                                       "--generations", "2", "--popsize", "50", "--seed", "1"]),
                   str(output))
    lines = output.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("gen,fevals"))
    assert [line.split(",")[0] for line in lines[header + 1:]] == ["1", "2"]
