"""Helpers of the example scripts (imported, not run)."""

import os
import pathlib
import re
import sys
import textwrap

import numpy as np

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


# Runs `f()` and copies everything that Julia writes to its stdout to the original
# stdout and to the file `path`, like `| tee path`.  The reader runs on another
# thread, so the output is shown while NeoGP runs.
_tee = jl.seval("""
function (f, path)
    out = stdout
    open(path, "w") do file
        pipe = Pipe()
        Base.link_pipe!(pipe; reader_supports_async = true, writer_supports_async = true)
        reader = Threads.@spawn for line in eachline(pipe)
            println(out, line)
            flush(out)
            println(file, line)
        end
        try
            redirect_stdout(() -> f(), pipe)
        finally
            close(pipe.in)
            wait(reader)
        end
    end
    nothing
end
""")


def run_neogp(args, output):
    """Run the command line interface of NeoGP (`script/runNeoGP.jl` of NeoGP.jl) with the
    arguments `args` and the extra arguments of the example script, writing its output to
    stdout and to the file `output`."""
    args = [str(a) for a in args] + sys.argv[1:]
    sys.stdout.flush()
    _tee(lambda: neogp.main(args), str(output))


def last_model_raw_expression(output):
    """The expression of the model in the last line of the output of NeoGP (the file
    `output`) as printed by NeoGP.  The expression is the last, quoted field of the line; it
    is followed by the name of the likelihood and the likelihood parameters, which are
    removed."""
    line = pathlib.Path(output).read_text(encoding="utf-8").splitlines()[-1]
    expr = re.sub(r'.*,"([^"]*)"$', r"\1", line)
    return re.sub(r" [^ ]+ (Float[0-9]+)?\[.*\]$", "", expr)


def last_model_title(output, width=110):
    """The expression of the model for the title of a plot, wrapped at spaces into lines of
    at most `width` characters."""
    return textwrap.fill(last_model_raw_expression(output), width)


# The functions of NeoGP's expressions (`inv` is printed as a division).
_functions = {name: getattr(np, name) for name in
              ("sqrt", "log", "exp", "abs", "sin", "cos", "tan", "sinh", "cosh", "tanh")}
_functions["sqr"] = np.square


def last_model_function(output, varnames):
    """The model in the last line of the output of NeoGP (the file `output`) as a numpy
    function of the variables `varnames`.  The expression is translated to Python:
      - Julia prints small `Float32` numbers as `1.0f-5`,
      - negative numbers get parentheses, because `**` binds stronger than the sign,
      - `^` is `**` in Python."""
    expr = last_model_raw_expression(output)
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
