"""ANI-1. DOI: 10.1039/c6sc05720a."""
import numpy as np
import chainer
import chainer.functions as F
from chainer.backend import get_array_module
from chmd.neighbors import duo_index, distance, distance_angle, neighbor_trios
from chmd.cutoffs import CosineCutoff


class ANI1AEV(object):
    """Compute Full AEV."""

    def __init__(self, num_elements, Rcr, Rca,
                 EtaR, ShfR, Zeta, ShfZ, EtaA, ShfA):
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
        self.Rcr = Rcr
        self.Rca = Rca
        self.radial = ANI1Radial(num_elements, EtaR, ShfR, Rcr)
        self.angular = ANI1Angular(num_elements, EtaA, Zeta, ShfA, ShfZ, Rca)

    def __call__(self, cells, ri, ei, i1, i2, j2, s2):
        """Calculate full AEV.

        Parameters
        ----------
        cells: (n_batch, 3, 3)
        ri: (n_atoms,)
        ei: (n_atoms,)
        i1: (n_atoms,)
        i2: (n_duos,)
        j2: (n_duos,)
        s2: (n_duos, 3)

        """
        xp = get_array_module(ri)
        rij_full = distance(cells, ri, i1, i2, j2, s2)
        in_rc_rad = rij_full.data < self.Rcr
        in_rc_ang = rij_full.data < self.Rca
        g_rad = self.radial(rij_full[in_rc_rad], ei,
                            i2[in_rc_rad], j2[in_rc_rad])
        i2_a = i2[in_rc_ang]
        j2_a = j2[in_rc_ang]
        s2_a = s2[in_rc_ang]
        i3_a, j3_a = neighbor_trios(i2_a, j2_a, xp)
        rij3, rik3, cosijk = distance_angle(
            cells, ri, i1, i2_a, j2_a, s2_a, i3_a, j3_a)
        g_ang = self.angular(rij3, rik3, cosijk, ei, i2_a, j2_a, i3_a, j3_a)
        return F.concat([g_rad, g_ang], axis=1)


class ANI1Radial(object):
    """Eq (3)."""

    def __init__(self, num_elements, EtaR, ShfR, Rcr):
        """Initializer.

        Parameters
        ----------
        num_elements: number of elements (species).
        EtaR: eta in eq (3).
        ShfR: Rs in eq (3).
        Rcr: cutoff radius of fc in eq (3).

        """
        self.num_elements = num_elements
        self.EtaR = EtaR
        self.ShfR = ShfR
        self.cutoff = CosineCutoff(Rcr)

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

    def __init__(self, num_elements, EtaA, Zeta, ShfA, ShfZ, Rca):
        """Initilizer.

        Parameters
        ----------
        num_elements: number of elements (species).
        EtaA: eta in eq (4).
        Zeta: zeta in eq(4).
        ShfA: Rs in eq(4).
        ShfZ: theta_s in eq(4).
        Rca: cutoff radius for eq (4).

        """
        xp = np
        self.num_elements = num_elements
        self.EtaA = EtaA
        self.Zeta = Zeta
        self.ShfA = ShfA
        self.ShfZ = ShfZ
        self.cutoff = CosineCutoff(Rca)
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

# def radial_terms(num_elements: int, EtaR: np.ndarray, ShfR: np.ndarray,
#                  ei: np.ndarray, i2: np.ndarray, j2: np.ndarray,
#                  rij: Variable, fc: Callable,
#                  ) -> np.ndarray:
#     """Eq (3).
#
#     Parameters
#     ----------
#     num_elements: number of elements.
#     ei: element number of each atom. (n_solo,)
#     i2: duo index. (n_duo,)
#     j2: duo index. (n_duo,)
#     rij: (n_duo,)
#     fc : fc(rij). (n_duo,)
#     EtaR: (n_eta, n_shf)
#     ShfR: (n_shf, n_shf)
#
#     Returns
#     -------
#     Gr: (n_solo, ne * n_eta * n_shf)
#
#     """
#     n_duo = rij.shape[0]
#     n_eta, n_shf = EtaR.shape
#     xp = get_array_module(rij)
#     n_solo = ei.shape[0]
#     assert ei.shape == (n_solo,)
#     assert rij.shape == (n_duo,)
#     assert EtaR.shape == (n_eta, n_shf)
#     assert ShfR.shape == (n_eta, n_shf)
#     # (n_duo, n_eta, n_shf)
#     r = rij[:, xp.newaxis, xp.newaxis]
#     f = fc(r)
#     shf = ShfR[xp.newaxis, :, :]
#     eta = EtaR[xp.newaxis, :, :]
#     peaks = (0.25 * F.exp(-eta * (r - shf) ** 2) * f)
#     flat_peaks = F.reshape(peaks, (n_duo, n_eta * n_shf))
#     seed = xp.zeros((n_solo * num_elements, n_eta * n_shf))
#     ej2 = ei[j2]
#     scattered = F.scatter_add(seed, i2 * num_elements + ej2, flat_peaks)
#     return scattered.reshape(n_solo, num_elements * n_eta * n_shf)
#
#
#
#
# def angular_terms(num_elements: int,
#                   EtaA: np.ndarray, Zeta: np.ndarray,
#                   ShfA: np.ndarray, ShfZ: np.ndarray,
#                   ei: np.ndarray, duo_index: np.ndarray,
#                   i2, j2, i3, j3,
#                   rij: Variable, rik: Variable, cosijk,
#                   fc: Callable,
#                   ):
#     n_eta_a, n_zeta, n_shf_a, n_shf_z = ShfA.shape
#     n_solo = ei.shape[0]
#     n_trio = rij.shape[0]
#     xp = rij.xp
#     assert rij.shape == (n_trio,)
#     assert rik.shape == (n_trio,)
#     assert cosijk.shape == (n_trio,)
#     assert EtaA.shape == (n_eta_a, n_zeta, n_shf_a, n_shf_z)
#     assert Zeta.shape == (n_eta_a, n_zeta, n_shf_a, n_shf_z)
#     assert ShfA.shape == (n_eta_a, n_zeta, n_shf_a, n_shf_z)
#     assert ShfZ.shape == (n_eta_a, n_zeta, n_shf_a, n_shf_z)
#     theta = F.arccos(cosijk * 0.95)
#     fcj = fc(rij)
#     fck = fc(rik)
#     rij = rij[:, None, None, None, None]
#     rik = rik[:, None, None, None, None]
#     fcj = fcj[:, None, None, None, None]
#     fck = fck[:, None, None, None, None]
#     theta = theta[:, None, None, None, None]
#     shf_z = ShfZ[None, :, :, :, :]
#     zeta = Zeta[None, :, :, :, :]
#     eta_a = EtaA[None, :, :, :, :]
#     shf_a = ShfA[None, :, :, :, :]
#     factor1 = ((1 + F.cos(theta - shf_z)) / 2) ** zeta
#     factor2 = F.exp(-eta_a * ((rij + rik) / 2 - shf_a) ** 2)
#     # (n_trio, n_eta_a, n_zeta, n_shf_a, n_shf_z)
#     peaks = 2 * factor1 * factor2 * fcj * fck
#     # (n_trio, n_eta_a * n_zeta * n_shf_a * n_shf_z)
#     n1 = n_eta_a * n_zeta * n_shf_a * n_shf_z
#     flat_peaks = F.reshape(peaks, (n_trio, n1))
#     numnum = num_elements * (num_elements + 1) // 2
#     seed = xp.zeros((n_solo * numnum, n1))
#     center = i2[i3]
#     ej3 = symmetric_duo_index(duo_index, xp=xp)[ei[j2[i3]], ei[j2[j3]]]
#
#     scattered = F.scatter_add(seed, center * numnum + ej3, flat_peaks)
#     return scattered.reshape(n_solo, numnum * n1)
#
#
