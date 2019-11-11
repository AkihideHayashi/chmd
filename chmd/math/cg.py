"""Conjugate Gradient."""
from typing import Callable
import numpy as np
import chainer


def linear_conjugate_gradient(ax_eval: Callable, b: np.ndarray, x0: np.ndarray,
                              tol: float):
    """Ordinary conjugate gradient.

    Solve Ax = b for x.

    Parameters
    ----------
    ax_eval: Evaluate ax_eval(x) == Ax
    b: Right side.
    x0: Initial x.
    tol: tolerance. ||Ax - b|| < tol.

    """
    xp = chainer.backend.get_array_module(b)
    xo = x0
    ro = ax_eval(xo) - b
    po = -ro
    while xp.linalg.norm(po) > tol:
        Apo = ax_eval(po)
        alpha = (ro @ ro) / (po @ Apo)
        xo = xo + alpha * po
        rn = ro + alpha * Apo
        beta = (rn @ rn) / (ro @ ro)
        po = - rn + beta * po
        ro = rn
    return xo
