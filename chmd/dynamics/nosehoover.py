"""Nose-Hoover thermostat."""
import numpy as np
import chainer
from chainer.backend import get_array_module
from chmd.math.xp import scatter_add_to_zero, repeat_interleave, cumsum_from_zero


def nose_hoover_accelerate(force_ext, v, m, thermostat, targets, kbt):
    """Calculate acceleration of hoover chain."""
    a_ext = force_ext / m
    xp = chainer.backend.get_array_module(a_ext)
    dtype = chainer.config.dtype
    shape = a_ext.shape
    ones = xp.ones_like(a_ext, dtype=dtype)
    control1 = scatter_add_to_zero(shape, thermostat, (m * v * v)[targets])
    control2 = -1 * scatter_add_to_zero(shape, thermostat, ones[targets]) * kbt
    controled = -1 * scatter_add_to_zero(shape, targets, v[thermostat]) * v
    return a_ext + (control1 + control2) / m + controled


def nose_hoover_scf(a_old, v_old, force_ext_new, m,
                    therm_number, therm_target, kbt, dt, tol):
    """Calculate self consistent acceleration.

    Returns
    -------
    v, a
    """
    a_new_old = force_ext_new / m
    while True:
        v_new = v_old + 0.5 * (a_old + a_new_old) * dt
        a_new_new = nose_hoover_accelerate(force_ext_new, v_new, m,
                                           therm_number, therm_target, kbt)
        if np.linalg.norm(a_new_new - a_new_old) < tol:
            return v_new, a_new_new
        a_new_old = a_new_new


def nose_hoover_conserve(x, v, m, therm_number, therm_target, kbt, pot, i1):
    """Calculate conserved value of hoover chain."""
    xp = chainer.backend.get_array_module(x)
    dtype = chainer.config.dtype
    ones = xp.ones_like(x, dtype=dtype)
    pot_therm = scatter_add_to_zero(
        x.shape, therm_number, ones[therm_target]) * kbt * x
    return scatter_add_to_zero(pot.shape, i1, 0.5 * v * v * m + pot_therm) + pot
    # return xp.sum(0.5 * v * v * m) + pot + np.sum(pot_therm)


def setup_nose_hoover_broadcast_atomic_part(thermostat_numbers,
                                            thermostat_targets,
                                            ndim):
    """Set up."""
    xp = get_array_module(thermostat_numbers)
    tt = thermostat_targets
    tn = thermostat_numbers
    atoms = tt[tt >= 0]
    therm = tt[tt < 0]
    flat_tt = xp.concatenate(
        [
            (
                xp.broadcast_to(
                    atoms[:, xp.newaxis],
                    (len(atoms), ndim)
                ) * ndim + xp.arange(ndim)[xp.newaxis, None]
            ).flatten(),
            therm
        ]
    )
    atoms = tn[tt >= 0]
    therm = tn[tt < 0]
    flat_tn = xp.concatenate(
        [xp.broadcast_to(atoms[:, xp.newaxis],
                         (len(atoms), ndim)).flatten(),
         therm])
    return flat_tn, flat_tt


def setup_nose_hoover_serial(x, v, m,
                             therm_number, therm_target, kbt, timeconst):
    """Set up parameters."""
    xp = get_array_module(x)
    n_free = len(x)
    therm_number = -1 - therm_number + n_free
    therm_target = xp.where(therm_target >= 0,
                            therm_target,
                            n_free - therm_target - 1)
    omega = 2 * xp.pi / timeconst
    therm_id, target_number = xp.unique(therm_number, return_counts=True)
    therm_x = xp.zeros(len(therm_id))
    therm_v = xp.zeros(len(therm_id))
    therm_m = target_number * kbt / omega / omega
    idx = xp.arange(len(x) + len(therm_id))
    is_atom = idx < len(x)
    return (xp.concatenate([x, therm_x]),
            xp.concatenate([v, therm_v]),
            xp.concatenate([m, therm_m]),
            therm_number,
            therm_target,
            xp.concatenate([xp.zeros_like(m), kbt]),
            is_atom,
            )


def setup_nose_hoover(x, v, m, i1, therm_number, therm_target, kbt, therm_time):
    xp = get_array_module(x[0])
    n = len(xp.unique(i1))
    x = [x[i1 == i] for i in range(n)]
    v = [v[i1 == i] for i in range(n)]
    m = [m[i1 == i] for i in range(n)]
    lst_x = []
    lst_v = []
    lst_m = []
    lst_tn = []
    lst_tt = []
    lst_kbt = []
    lst_ia = []
    for i, _ in enumerate(x):
        ndim = x[i].shape[1]
        tn, tt = setup_nose_hoover_broadcast_atomic_part(
            therm_number[i], therm_target[i], ndim)
        xi, vi, mi, tni, tti, kbti, iai = setup_nose_hoover_serial(
            x[i].flatten(), v[i].flatten(),
            xp.broadcast_to(m[i][:, None], x[i].shape).flatten(), tn, tt,
            kbt[i], therm_time[i])
        lst_x.append(xi)
        lst_v.append(vi)
        lst_m.append(mi)
        lst_tn.append(tni)
        lst_tt.append(tti)
        lst_kbt.append(kbti)
        lst_ia.append(iai)
    positions = np.concatenate(lst_x)
    velocities = np.concatenate(lst_v)
    masses = np.concatenate(lst_m)
    n_thermstat = xp.array([len(xi) for xi in lst_tn])
    n_free = xp.array([len(xi) for xi in lst_x])
    pad = cumsum_from_zero(n_free)[repeat_interleave(n_thermstat)]
    m_therm_number = xp.concatenate(lst_tn) + pad
    m_therm_target = xp.concatenate(lst_tt) + pad
    m_kbt = xp.concatenate(lst_kbt)
    is_atom = xp.concatenate(lst_ia)
    extended_i1 = repeat_interleave(n_free)
    return (positions, velocities, masses,
            m_therm_number, m_therm_target, m_kbt, is_atom, extended_i1)
