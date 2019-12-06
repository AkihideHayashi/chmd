"""Nose-Hoover thermostat."""
import numpy as np
import chainer
from chainer.backend import get_array_module
from chmd.math.xp import scatter_add_to_zero, repeat_interleave, cumsum_from_zero
from chmd.utils.batchform import parallel_form, series_form


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

def transform_conventional_thermostat(numbers, targets, natom, ndim):
    """Transform numbers and targets from conventional form.
    
    Parameters
    ----------
    numbers: Thermostat id. Negative numbers.
    targets: Target id. Positive is atom. Negative is thermostat.
    natom: number of atoms.
    ndim: number of dimension.
    
    Returns
    -------
    numbers: Thermostat id. Appended after atoms. Numbers in flatten shape.
    targets: Target id.
    
    Examples
    --------
    >>> numbers = [-1, -1, -1, -2, -3, -3]
    >>> targets = [ 0,  1,  2, -1, -2,  0]
    >>> transform_conventional_thermostat(numbers, targets, 4, 3)
        (array([12, 12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 15, 15, 15, 13, 14]),
         array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  0,  1,  2,  3,  4,  5, 12, 13]))
    """
    xp = np
    numbers = np.array(numbers)
    targets = np.array(targets)
    assert np.all(numbers < 0), "Thermostat numbers must be negative."
    to_atom = targets >= 0
    to_ther = targets < 0
    targets_to_atom = (targets[to_atom][:, xp.newaxis] * ndim) + xp.arange(ndim)[xp.newaxis, :]
    numbers_to_atom = -xp.broadcast_to(numbers[to_atom][:, xp.newaxis], targets_to_atom.shape) + natom * ndim - 1
    targets_to_atom = targets_to_atom.flatten()
    numbers_to_atom = numbers_to_atom.flatten()
    targets_to_ther = -targets[to_ther] + natom * ndim - 1
    numbers_to_ther = -numbers[to_ther] + natom * ndim - 1
    targets_ret = xp.concatenate([targets_to_atom, targets_to_ther])
    numbers_ret = xp.concatenate([numbers_to_atom, numbers_to_ther])
    return numbers_ret, targets_ret

def thermostat_mass(numbers, targets, timeconst, kbt):
    """Calculate appropriate thermostat mass.
    omega = 2pi / kbt
    mass = n_target_freedom * kbt / (omega * omega)
    """
    xp = get_array_module(numbers)
    ids, counts = np.unique(numbers, return_counts=True)
    omega = 2 * xp.pi / xp.array(timeconst)
    mass = counts * kbt / (omega * omega)
    return mass

def expand(x, shape, default=0.0):
    """Expand array in shape."""
    xp = get_array_module(x)
    new = xp.full(shape, default, x.dtype)
    n = x.shape[0]
    new[:n] = x
    return new

def setup_nose_hoover_chain_parallel(symbols, positions, velocities, masses,
                                     numbers, targets, timeconsts, kbts):
    (sym, pos, vel, mas, nu, ta, kbt, isa, ist) = zip(*[
        setup_nose_hoover_chain(sym, pos, vel, mas, nu, ta, tc, kt)
        for sym, pos, vel, mas, nu, ta, tc, kt in zip(
            symbols, positions, velocities, masses,
            numbers, targets, timeconsts, kbts)])
    (sym, pos, vel, mas, kbt, isa, ist), _ = parallel_form.from_list(
        [sym, pos, vel, mas, kbt, isa, ist],
        ['', 0.0, 0.0, 1.0, 0.0, False, False]
    )
    _, n_atoms, n_dim = pos.shape
    (nu, ta), affiliation = series_form.from_list([nu, ta])
    nu = nu + affiliation * n_atoms * n_dim
    ta = ta + affiliation * n_atoms * n_dim
    return sym, pos, vel, mas, nu, ta, kbt, isa, ist

def setup_nose_hoover_chain(symbols, positions, velocities, masses, numbers, targets, timeconst, kbt):
    """Transform all about Nose-Hoover.
    
    Parameters
    ----------
    symbols: str[atoms,]
    positions: float[atoms, dim]
    velocities: float[atoms, dim]
    masses: float[atoms]
    numbers: Thermostat id. Negative numbers.
    targets: Target id. Positive is atom. Negative is thermostat.
    timeconst: float[n_therm]
    kbt: float[n_therm]. kB * T for each thermostat.
    
    Returns
    -------
    symbols: str[expanded]
    positions: float[expanded, dim]
    velocities: float[expanded, dim]
    masses: float[expanded, dim]
    numbers: Thermostat id. Appended after atoms. Numbers in flatten shape.
    targets: Target id.
    kbt: float[expanded, dim]
    is_atom[expanded]
    is_therm[expanded, dim]
    """
    if isinstance(positions, list):
        positions = np.array(positions)
    if isinstance(velocities, list):
        velocities = np.array(velocities)
    if isinstance(symbols, list):
        symbols = np.array(symbols)
    if isinstance(masses, list):
        masses = np.array(masses)
    xp = get_array_module(positions)
    natom, ndim = positions.shape
    nther = len(timeconst)
    r_numbers, r_targets = transform_conventional_thermostat(numbers, targets, natom, ndim)
    t_mass = thermostat_mass(r_numbers, r_targets, timeconst, kbt)
    n_append = (nther - 1) // ndim + 1
    r_symbols = expand(symbols, (natom + n_append,), '')
    r_positions = expand(positions, (natom + n_append, ndim))
    r_velocities = expand(velocities, (natom + n_append, ndim))
    is_atom = xp.arange(r_positions.shape[0]) < natom
    is_therm = xp.full(r_positions.shape, False).flatten()
    is_therm[r_numbers] = True
    is_therm = is_therm.reshape(r_positions.shape)
    r_masses = xp.full((natom + n_append, ndim), 1.0)
    r_masses[:natom, :] = masses[:, xp.newaxis]
    r_masses[is_therm] = t_mass
    r_kbt = xp.zeros_like(r_positions)
    r_kbt[is_therm] = kbt
    return r_symbols, r_positions, r_velocities, r_masses, r_numbers, r_targets, r_kbt, is_atom, is_therm

# def setup_nose_hoover_broadcast_atomic_part(thermostat_numbers,
#                                             thermostat_targets,
#                                             ndim):
#     """Set up."""
#     xp = get_array_module(thermostat_numbers)
#     tt = thermostat_targets
#     tn = thermostat_numbers
#     atoms = tt[tt >= 0]
#     therm = tt[tt < 0]
#     flat_tt = xp.concatenate(
#         [
#             (
#                 xp.broadcast_to(
#                     atoms[:, xp.newaxis],
#                     (len(atoms), ndim)
#                 ) * ndim + xp.arange(ndim)[xp.newaxis, None]
#             ).flatten(),
#             therm
#         ]
#     )
#     atoms = tn[tt >= 0]
#     therm = tn[tt < 0]
#     flat_tn = xp.concatenate(
#         [xp.broadcast_to(atoms[:, xp.newaxis],
#                          (len(atoms), ndim)).flatten(),
#          therm])
#     return flat_tn, flat_tt


# def setup_nose_hoover_serial(x, v, m,
#                              therm_number, therm_target, kbt, timeconst):
#     """Set up parameters."""
#     xp = get_array_module(x)
#     n_free = len(x)
#     therm_number = -1 - therm_number + n_free
#     therm_target = xp.where(therm_target >= 0,
#                             therm_target,
#                             n_free - therm_target - 1)
#     omega = 2 * xp.pi / timeconst
#     therm_id, target_number = xp.unique(therm_number, return_counts=True)
#     therm_x = xp.zeros(len(therm_id))
#     therm_v = xp.zeros(len(therm_id))
#     therm_m = target_number * kbt / omega / omega
#     idx = xp.arange(len(x) + len(therm_id))
#     is_atom = idx < len(x)
#     return (xp.concatenate([x, therm_x]),
#             xp.concatenate([v, therm_v]),
#             xp.concatenate([m, therm_m]),
#             therm_number,
#             therm_target,
#             xp.concatenate([xp.zeros_like(m), kbt]),
#             is_atom,
#             )


# def setup_nose_hoover(x, v, m, i1, therm_number, therm_target, kbt, therm_time):
#     xp = get_array_module(x[0])
#     n = len(xp.unique(i1))
#     x = [x[i1 == i] for i in range(n)]
#     v = [v[i1 == i] for i in range(n)]
#     m = [m[i1 == i] for i in range(n)]
#     lst_x = []
#     lst_v = []
#     lst_m = []
#     lst_tn = []
#     lst_tt = []
#     lst_kbt = []
#     lst_ia = []
#     for i, _ in enumerate(x):
#         ndim = x[i].shape[1]
#         tn, tt = setup_nose_hoover_broadcast_atomic_part(
#             therm_number[i], therm_target[i], ndim)
#         xi, vi, mi, tni, tti, kbti, iai = setup_nose_hoover_serial(
#             x[i].flatten(), v[i].flatten(),
#             xp.broadcast_to(m[i][:, None], x[i].shape).flatten(), tn, tt,
#             kbt[i], therm_time[i])
#         lst_x.append(xi)
#         lst_v.append(vi)
#         lst_m.append(mi)
#         lst_tn.append(tni)
#         lst_tt.append(tti)
#         lst_kbt.append(kbti)
#         lst_ia.append(iai)
#     positions = np.concatenate(lst_x)
#     velocities = np.concatenate(lst_v)
#     masses = np.concatenate(lst_m)
#     n_thermstat = xp.array([len(xi) for xi in lst_tn])
#     n_free = xp.array([len(xi) for xi in lst_x])
#     pad = cumsum_from_zero(n_free)[repeat_interleave(n_thermstat)]
#     m_therm_number = xp.concatenate(lst_tn) + pad
#     m_therm_target = xp.concatenate(lst_tt) + pad
#     m_kbt = xp.concatenate(lst_kbt)
#     is_atom = xp.concatenate(lst_ia)
#     extended_i1 = repeat_interleave(n_free)
#     return (positions, velocities, masses,
#             m_therm_number, m_therm_target, m_kbt, is_atom, extended_i1)
