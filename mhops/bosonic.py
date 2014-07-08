#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
import scipy.sparse as sparse

from hops.hstruct import HierarchyStructure
from ._base import MasterIntegrator, commutator, multiply_raveled, complex_t


class BosonicIntegrator(MasterIntegrator):

    """Docstring for BosonicIntegrator. """

    def __init__(self, depth, bath, h_sys):
        """@todo: to be defined1. """
        MasterIntegrator.__init__(self, bath, h_sys)

        struct = HierarchyStructure(2 * self._modes, depth)
        self._nr_aux_states = struct.entries
        self._prop = self._setup_propagator(struct)

    def _setup_propagator(self, struct):
        """@todo: Docstring for _setup_propagator.

        struct -- TODO
        Returns:

        """
        # TODO THIS ASSUMES THAT INVALID_INDEX < 0; INCORPORATE INVALID_INDEX
        prop = sparse.lil_matrix((self.dim_aux, self.dim_aux), dtype=complex_t)
        dim_rho = self.dim_hs**2
        w = self._gamma + 1.j * self._omega

        for iind in xrange(self.nr_aux_states):
            base = iind * dim_rho
            m = struct.vecind[iind, :self._modes]
            n = struct.vecind[iind, self._modes:]
            prop[base:base + dim_rho, base:base + dim_rho] += \
                -1.j * commutator(self._h_sys) \
                - (np.dot(m, w) + np.dot(n, np.conj(w))) * np.identity(dim_rho)

            for mode in xrange(self._modes):
                couplop = self._l[self._l_map[mode]]

                base_c = struct.indbl[iind, mode] * dim_rho
                if base_c >= 0:
                    prop[base:base + dim_rho, base_c:base_c + dim_rho] += \
                        multiply_raveled(m[mode] * self._g[mode] * couplop,
                                         side='l')

                base_c = struct.indbl[iind, self._modes + mode] * dim_rho
                if base_c >= 0:
                    prop[base:base + dim_rho, base_c:base_c + dim_rho] += \
                        multiply_raveled(n[mode] * np.conj(self._g[mode]) *
                                         np.conj(couplop.T), side='r')

                base_c = struct.indab[iind, mode] * dim_rho
                if base_c >= 0:
                    prop[base:base + dim_rho, base_c:base_c + dim_rho] += \
                        - commutator(np.conj(couplop.T))

                base_c = struct.indab[iind, self._modes + mode] * dim_rho
                if base_c >= 0:
                    prop[base:base + dim_rho, base_c:base_c + dim_rho] += \
                        commutator(couplop)

        return prop.tocsr()


def simple_bosonic_meq(depth, g, gamma, omega, h_sys):
    """Spectrum hierarchy with identical, independent baths with bcf
            alpha_i(t) = sum_j g_j * exp(-gamma_j|t| - ii * Omega_j * t)

    depth          -- depth of the hierarchy
    g[modes]       -- coupling strengths of the modes
    gamma[modes]   -- damping sof the modes
    omega[modes]   -- central frequencies of the modes
    h_sys[dim,dim] -- system hamiltonian

    Usage:
    meq = master.simple_bosonic_meq(Depth, g, gamma, Omega, h)
    dt = tLength / (tSteps - 1)
    t, rho = meq.get_rho(dt, tLength)

    """
    dim = np.shape(h_sys)[0]
    bath = {'g': [g for _ in range(dim)],
            'gamma': [gamma for _ in range(dim)],
            'Omega': [omega for _ in range(dim)]}

    return BosonicIntegrator(depth, bath, h_sys)
