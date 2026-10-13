"""
The Julia module NeoGP, imported with ``from pyneogp import neogp``.

This module replaces itself with the Julia module, so the Julia
API is used from Python as it is used from Julia: ``neogp.NSGA2(...)``,
``neogp.evolve_b(gp, 100)`` (``!`` becomes ``_b``), ``neogp.ADD``, ...

In addition, the Julia functions and types are wrapped so that their arguments
are converted to the Julia types that NeoGP expects (juliacall passes numpy
arrays and Python lists as views, which NeoGP's methods do not accept):

- numpy arrays (and objects with ``__array__``, e.g. pandas columns) of
  numbers are copied into Julia ``Array``\\ s of the same element type, so
  ``float32`` data give ``Float32`` models,
- lists become Julia vectors with the narrowest element type, e.g. a list of
  ``str`` a ``Vector{String}``,
- tuples and lists are converted element by element, and wrapped Julia
  functions are unwrapped, e.g. ``(neogp.negloglik, neogp.individual_length)``.

Return values are not converted; Julia vectors returned to Python support
``len``, indexing, iteration and ``numpy.asarray``.

The fitness of individuals is evaluated on Julia worker threads, which must not
call Python, so objectives (``indiv_loss_func``) must be Julia functions.
Callbacks that run on the main thread, such as ``iter_callback`` of
``evolve_b``, can be Python functions.
"""

import sys
import types

import numpy as np

from . import load_julia_packages

_NeoGP = load_julia_packages("NeoGP")

from juliacall import AnyValue, Main as _jl  # noqa: E402  (after Julia is set up)

_to_array = _jl.seval("x -> Array(x)")
_narrow = _jl.seval("v -> isempty(v) ? Any[] : identity.(collect(v))")
_is_callable = _jl.seval("x -> x isa Function || x isa Type")


def to_julia(x):
    """Convert a Python argument to the Julia value passed to NeoGP (see the module docstring)."""
    if isinstance(x, JuliaFunction):
        return x.julia
    if isinstance(x, AnyValue):  # a Julia value, passed on unchanged
        return x
    if isinstance(x, tuple):
        return tuple(to_julia(e) for e in x)
    if isinstance(x, list):
        return _narrow([to_julia(e) for e in x])
    if isinstance(x, np.generic):  # numpy scalar
        return x.item()
    if isinstance(x, np.ndarray) or (hasattr(x, "__array__") and not isinstance(x, (str, bytes))):
        a = np.asarray(x)
        if a.dtype.kind in "biuf":
            return _to_array(a)
        return _narrow([to_julia(e) for e in a.tolist()])
    return x


class JuliaFunction:
    """A Julia function or type of NeoGP that converts its arguments with :func:`to_julia`."""

    __slots__ = ("julia",)

    def __init__(self, julia):
        self.julia = julia

    def __call__(self, *args, **kwargs):
        return self.julia(*(to_julia(a) for a in args),
                          **{k: to_julia(v) for k, v in kwargs.items()})

    def __getitem__(self, key):  # type parameters, e.g. neogp.Dataset[jl.Float32]
        return self.julia[to_julia(key)]

    def __getattr__(self, name):
        return getattr(self.julia, name)

    def __repr__(self):
        return repr(self.julia)

    @property
    def __doc__(self):
        return str(_jl.Docs.doc(self.julia))


class _NeoGPModule(types.ModuleType):
    """The Julia module NeoGP, with its functions and types wrapped in :class:`JuliaFunction`."""

    def __init__(self, julia_module):
        super().__init__(__name__, __doc__)
        self.__dict__.update(julia_module=julia_module, to_julia=to_julia,
                             JuliaFunction=JuliaFunction, __file__=__file__,
                             __package__=__package__)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = getattr(self.julia_module, name)  # translates `_b` to `!`
        if isinstance(value, AnyValue) and _is_callable(value):
            value = JuliaFunction(value)
        self.__dict__[name] = value  # cache
        return value

    def __dir__(self):
        names = (str(n) for n in _jl.names(self.julia_module))
        return sorted(set(super().__dir__()) | {n[:-1] + "_b" if n.endswith("!") else n for n in names})


sys.modules[__name__] = _NeoGPModule(_NeoGP)  # mutate myself
