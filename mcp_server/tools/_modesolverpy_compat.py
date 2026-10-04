"""Compatibility shims for modesolverpy 0.4.4 on current NumPy/SciPy.

modesolverpy has not been released since 2019 and relies on APIs that were
removed upstream: the ``'complex_'`` dtype alias and ``numpy.trapz`` (NumPy
2.0), ``scipy.sqrt`` (SciPy 1.12) and ``scipy.interpolate.interp2d`` (SciPy
1.14). Pinning around them is not an option because gdsfactory's kfactory
dependency requires SciPy >= 1.14, so ``install()`` restores just those names.

It also registers no-op stand-ins for modesolverpy's optional plotting and
material helpers (gnuplotpy, opticalmaterialspy) when they are not installed:
modesolverpy imports them at module level, but the eigenmode solver itself
never calls them.
"""
import importlib
import sys
import types
import warnings

import numpy as np
import scipy
import scipy.interpolate

_OPTIONAL_MODULES = ("gnuplotpy", "opticalmaterialspy")


class _Interp2d:
    """Minimal drop-in for the removed ``scipy.interpolate.interp2d``.

    Covers the call pattern modesolverpy uses: build from a regular (x, y)
    grid with z shaped (len(y), len(x)), then evaluate on 1-D x and y vectors
    to get an array shaped (len(y), len(x)).
    """

    _DEGREE = {"linear": 1, "cubic": 3, "quintic": 5}

    def __init__(self, x, y, z, kind="linear", copy=True, bounds_error=False, fill_value=None):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        z = np.asarray(z)
        if z.ndim == 1:
            z = z.reshape(y.size, x.size)
        degree = self._DEGREE[kind]
        spline = scipy.interpolate.RectBivariateSpline
        self._real = spline(y, x, z.real, kx=degree, ky=degree)
        self._imag = spline(y, x, z.imag, kx=degree, ky=degree) if np.iscomplexobj(z) else None

    def __call__(self, x, y, dx=0, dy=0, assume_sorted=False):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        # RectBivariateSpline's first axis is our y, so the derivative orders swap.
        out = self._real(y, x, dx=dy, dy=dx)
        if self._imag is not None:
            out = out + 1j * self._imag(y, x, dx=dy, dy=dx)
        return out


class _Sink:
    """Accepts any attribute access or call; stands in for plotting helpers."""

    def __call__(self, *args, **kwargs):
        return _Sink()

    def __getattr__(self, name):
        return _Sink()


def _interp2d_usable() -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scipy.interpolate.interp2d([0.0, 1.0], [0.0, 1.0], [[0.0, 0.0], [0.0, 0.0]])
        return True
    except Exception:
        return False


def _stub_missing_optional_modules() -> None:
    for name in _OPTIONAL_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            module = types.ModuleType(name)
            module.__getattr__ = lambda attr: _Sink()
            sys.modules[name] = module


_installed = False


def install() -> None:
    """Idempotently apply the shims. Must run before modesolverpy is imported."""
    global _installed
    if _installed:
        return
    np.sctypeDict.setdefault("complex_", np.complex128)
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    if not hasattr(scipy, "sqrt"):
        scipy.sqrt = np.sqrt
    if not _interp2d_usable():
        scipy.interpolate.interp2d = _Interp2d
    _stub_missing_optional_modules()
    _installed = True
