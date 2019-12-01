"""ANI-1. DOI: 10.1039/c6sc05720a."""
import numpy as np
import chainer
import chainer.functions as F
from chainer.backend import get_array_module
from chmd.functions.neighbors import (duo_index, distance,
                                      distance_angle, neighbor_trios)
from chmd.functions.cutoffs import CosineCutoff


class ANI1AEV(object):
    """Compute Full AEV."""

    def __init__(self, num_elements, radial, angular):
        """Initializer.

        Parameters
        ----------
        num_elements: number of elements (species).
        Rcr: parameter of ANI1Radial
        Rca: parameter of ANI1Angular
        EtaR: parameter of ANI1Radial
        ShfR: parameter of ANI1Radial
        Zeta: parameter of ANI1Angular
        ShfZ: parameter of ANI1Angular
        EtaA: parameter of ANI1Angular
        ShfA: parameter of ANI1Angular

        """
        self.num_elements = num_elements
        self.radial = ANI1Radial(num_elements, **radial)
        self.angular = ANI1Angular(num_elements, **angular)

    def __call__(self, ri, ei, i2, j2, s2):
        """Calculate full AEV. Inputs are serial form or flatten form.

        Parameters
        ----------
        ri: (n_atoms, dim) cartesian
        ei: (n_atoms,)
        i2: (n_duos,)
        j2: (n_duos,)
        s2: (n_duos, dim) cartesian

        """
        rij_full = distance(ri, i2, j2, s2)
        in_rc_rad = rij_full.data < self.radial.rc
        in_rc_ang = rij_full.data < self.angular.rc

        rij_r = rij_full[in_rc_rad]
        i2_r = i2[in_rc_rad]
        j2_r = j2[in_rc_rad]
        g_rad = self.radial(rij_r, ei, i2_r, j2_r)

        i2_a = i2[in_rc_ang]
        j2_a = j2[in_rc_ang]
        s2_a = s2[in_rc_ang]
        i3_a, j3_a = neighbor_trios(i2_a, j2_a)
        rij3, rik3, cosijk = distance_angle(ri, i2_a, j2_a, s2_a, i3_a, j3_a)
        g_ang = self.angular(rij3, rik3, cosijk, ei, i2_a, j2_a, i3_a, j3_a)
        return F.concat([g_rad, g_ang], axis=1)


class ANI1Radial(object):
    """Eq (3)."""

    def __init__(self, num_elements, cutoff, head, tail, step, sigma):
        """Initializer.

        Parameters
        ----------
        num_elements: number of elements (species).
        EtaR: Define eta in eq (3). eta == 1 / (sigma * sigma)
        head, tail, step: Define Rs in eq (3). Define ShfR in torchani
        cutoff: cutoff radius of fc in eq (3). Rcr in torchani

        """
        self.rc = cutoff
        self.num_elements = num_elements
        self.EtaR = np.array([1.0 / (sigma * sigma)])
        self.ShfR = np.arange(head, tail, step)
        self.cutoff = CosineCutoff(cutoff)

    def __call__(self, rij, ei, i2, j2):
        """Calculate radial aev.

        Parameters
        ----------
        rij: (n_duo,)
        ei: (n_solo,)
        i2: (n_duo,)
        j2: (n_duo,)

        """
        xp = get_array_module(rij)
        dtype = chainer.config.dtype
        num_elements = self.num_elements
        n_duo = rij.shape[0]
        n_eta = self.EtaR.shape[0]
        n_shf = self.ShfR.shape[0]
        n_solo = ei.shape[0]
        assert ei.shape == (n_solo,)
        assert rij.shape == (n_duo,)
        assert self.EtaR.shape == (n_eta, )
        assert self.ShfR.shape == (n_shf, )
        # (n_duo, n_eta, n_shf)
        r = rij[:, xp.newaxis, xp.newaxis]
        f = self.cutoff(r)
        eta = xp.array(self.EtaR[xp.newaxis, :, xp.newaxis])
        shf = xp.array(self.ShfR[xp.newaxis, xp.newaxis, :])
        peaks = (0.25 * F.exp(-eta * (r - shf) ** 2) * f)
        flat_peaks = F.reshape(peaks, (n_duo, n_eta * n_shf))
        seed = xp.zeros((n_solo * num_elements, n_eta * n_shf), dtype=dtype)
        ej2 = ei[j2]
        scattered = F.scatter_add(seed, i2 * num_elements + ej2, flat_peaks)
        return scattered.reshape(n_solo, num_elements * n_eta * n_shf)


def symmetric_duo_index(di: np.ndarray, xp=np):
    """Auxiliary function for ANI1Angular.

    Calculate packed, symmetric duo index matrix.
    """
    symmetrix_di = np.min([di, di.T], axis=0)
    unique, inverse = xp.unique(symmetrix_di, return_inverse=True)
    return np.arange(unique.max())[inverse].reshape(di.shape)


class ANI1Angular(object):
    """Eq (4)."""

    def __init__(self, num_elements, cutoff, head, tail, step,
                 sigma, zeta, ndiv):
        """Initilizer.

        Parameters
        ----------
        num_elements: number of elements (species).
        cutoff: Rca in eq (4) or torchani.
        sigma: Define eta in eq (4) or EtaA in torchani.
        zeta: Zeta in eq(4) or torchani.
        head, tail, step: Define Rs in eq(4) or ShfA in torchani.
        ndiv: Define theta_s in eq(4) or ShfZ in torchani.

        """
        xp = np
        self.rc = cutoff
        self.num_elements = num_elements
        self.EtaA = np.array([1.0 / (sigma * sigma)])
        self.Zeta = np.array([zeta])
        self.ShfA = np.arange(head, tail, step)
        self.ShfZ = np.linspace(np.pi, 0, ndiv, endpoint=False)[::-1]
        self.cutoff = CosineCutoff(cutoff)
        self.symmetric_duo = symmetric_duo_index(
            duo_index(num_elements, xp), xp)

    def __call__(self, rij, rik, cosijk, ei, i2, j2, i3, j3):
        """Calculate angular aev.

        Parameters
        ----------
        rij: |ri - rj| (n_duo,)
        rik: |ri - rk| (n_duo,)
        cosijk: cos between rij and rik (n_duo,)
        ei: elements of each atom. (n_solo,)
        i2: (n_duo,)
        j2: (n_duo,)
        i3: (n_trio,)
        j3: (n_trio,)

        """
        dtype = chainer.config.dtype
        n_shf_a = self.ShfA.shape[0]
        n_eta_a = self.EtaA.shape[0]
        n_zeta = self.Zeta.shape[0]
        n_shf_z = self.ShfZ.shape[0]
        n_solo = ei.shape[0]
        n_trio = rij.shape[0]
        xp = get_array_module(rij)
        assert rij.shape == (n_trio,)
        assert rik.shape == (n_trio,)
        assert cosijk.shape == (n_trio,)
        assert self.EtaA.shape == (n_eta_a, )
        assert self.Zeta.shape == (n_zeta, )
        assert self.ShfA.shape == (n_shf_a, )
        assert self.ShfZ.shape == (n_shf_z, )
        theta = F.arccos(cosijk * 0.95)
        fcj = self.cutoff(rij)
        fck = self.cutoff(rik)
        rij = rij[:, None, None, None, None]
        rik = rik[:, None, None, None, None]
        fcj = fcj[:, None, None, None, None]
        fck = fck[:, None, None, None, None]
        theta = theta[:, None, None, None, None]
        eta_a = xp.array(self.EtaA[None, :, None, None, None])
        zeta = xp.array(self.Zeta[None, None, :, None, None])
        shf_a = xp.array(self.ShfA[None, None, None, :, None])
        shf_z = xp.array(self.ShfZ[None, None, None, None, :])
        factor1 = ((1 + F.cos(theta - shf_z)) / 2) ** zeta
        factor2 = F.exp(-eta_a * ((rij + rik) / 2 - shf_a) ** 2)
        # (n_trio, n_eta_a, n_zeta, n_shf_a, n_shf_z)
        peaks = 2 * factor1 * factor2 * fcj * fck
        # (n_trio, n_eta_a * n_zeta * n_shf_a * n_shf_z)
        n1 = n_eta_a * n_zeta * n_shf_a * n_shf_z
        flat_peaks = F.reshape(peaks, (n_trio, n1))
        numnum = self.num_elements * (self.num_elements + 1) // 2
        seed = xp.zeros((n_solo * numnum, n1), dtype=dtype)
        center = i2[i3]
        ej3 = xp.array(self.symmetric_duo)[ei[j2[i3]], ei[j2[j3]]]

        scattered = F.scatter_add(seed, center * numnum + ej3, flat_peaks)
        return scattered.reshape(n_solo, numnum * n1) / 2
