"""
Python bindings to NeoGP.jl, tree-based genetic programming for symbolic regression.

Julia and NeoGP.jl (declared in ``juliapkg.json``) are installed automatically by 
juliacall on first use, and

    from pyneogp import neogp

returns the Julia module NeoGP with Python-friendly argument conversion (see
``pyneogp.neogp``).  Julia functions ending in ``!`` are called with the suffix
``_b`` instead, e.g. ``neogp.evolve_b(gp, 100)`` for ``evolve!(gp, 100)``.
"""

import os

__version__ = "0.1.0"

# NeoGP evaluates the fitness of individuals on all Julia threads.  juliacall
# starts Julia single-threaded unless told otherwise, so default to all cores.
# Julia must handle signals when it runs several threads (see the juliacall
# docs).  These must be set before juliacall is imported, so they are set here
# and juliacall is only imported in `load_julia_packages`.
if "PYTHON_JULIACALL_THREADS" not in os.environ and "JULIA_NUM_THREADS" not in os.environ:
    os.environ["PYTHON_JULIACALL_THREADS"] = "auto"
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
# juliacall loads ~/.julia/config/startup.jl, which fails if it loads packages that are
# not installed for the Julia version of juliacall (e.g. Revise), so skip it by default.
os.environ.setdefault("PYTHON_JULIACALL_STARTUP_FILE", "no")


def load_julia_packages(*names):
    """
    Load the Julia packages ``names`` and return references to them (a single
    module for a single name, otherwise a tuple).

    juliacall installs Julia and the packages declared in ``pyneogp/juliapkg.json``
    (NeoGP and its dependencies) on first use, so they are importable here.
    Packages are not added on demand because NeoGP and its dependencies are not 
    in the General registry; add other packages with ``juliapkg.add`` (see the 
    juliapkg documentation).
    """
    from juliacall import Main

    Main.seval("import " + ", ".join(names))
    modules = tuple(Main.seval(name) for name in names)
    return modules[0] if len(modules) == 1 else modules
