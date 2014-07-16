#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
import scipy.sparse as sp

from hops.hstruct import HierarchyStructure
from ._base import MasterIntegrator, commutator, multiply_raveled, complex_t


class FermionicIntegrator(MasterIntegrator):

    """Docstring for FermionicIntegrator. """

    def __init__(self, bath, h_sys, couplops, pop_modes=None):
        """@todo: to be defined1.

        :param bath: @todo
        :param h_sys: @todo
        :param pop_modes: @todo

        """
        MasterIntegrator.__init__(self, bath, h_sys, couplops)

        self._pop_modes = pop_modes if pop_modes is not None else self._modes
        struct = HierarchyStructure(2 * self._modes, 1, mode='quadratic',
                                    pop_modes=2 * self._pop_modes)
        self._nr_aux_states = struct.entries
        self._prop = self._setup_propagator(struct)

        # print("WRONG")
        # print(self._prop)

    def _setup_propagator(self, struct):
        """@todo: Docstring for _setup_propagator.

        :param struct: @todo
        :returns: @todo

        """
        # TODO THIS ASSUMES THAT INVALID_INDEX < 0; INCORPORATE INVALID_INDEX
        prop = sp.lil_matrix((self.dim_aux, self.dim_aux), dtype=complex_t)
        dim_rho = self.dim_hs**2
        w = self._gamma + 1.j * self._omega
        adj = lambda A: np.conj(A.T)

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
                    pref = (-1)**(sum(m[mode+1:])) * self._g[mode]
                    prop[base:base + dim_rho, base_c:base_c + dim_rho] += \
                        multiply_raveled(pref * couplop, side='l')

                base_c = struct.indbl[iind, self._modes + mode] * dim_rho
                if base_c >= 0:
                    pref = (-1)**(sum(n[mode+1:])) * np.conj(self._g[mode])
                    prop[base:base + dim_rho, base_c:base_c + dim_rho] += \
                        multiply_raveled(pref * adj(couplop), side='r')

                base_c = struct.indab[iind, mode] * dim_rho
                if base_c >= 0:
                    prop[base:base + dim_rho, base_c:base_c + dim_rho] += \
                        - multiply_raveled((-1)**(sum(m[mode+1:])) * adj(couplop), side='l') \
                        + multiply_raveled((-1)**sum(n) * adj(couplop), side='r')

                base_c = struct.indab[iind, self._modes + mode] * dim_rho
                if base_c >= 0:
                    prop[base:base + dim_rho, base_c:base_c + dim_rho] += \
                        multiply_raveled((-1)**(sum(m)) * couplop, side='l') \
                        - multiply_raveled((-1)**sum(n[mode+1:]) * couplop,
                                           side='r')

        return prop.tocsr()


def _testcase_one_mode(h_sys, rho0, g, w, L, timesteps):
    """Integration of the single environment-mode case for debugging. The exact
    reduced dynamics is described by the reduced density operator p00 as well
    as three auxiliary states (p01, p10, p11). Their equations of motion read


    :param h_sys: @todo
    :param rho0: @todo
    :param g: @todo
    :param w: @todo
    :param L: @todo
    :param timesteps: @todo
    :returns: @todo

    """
    from numpy import dot
    from scipy.integrate import complex_ode

    dim = h_sys.shape[0]
    comm = lambda A, B: dot(A, B) - dot(B, A)
    adj = lambda A: np.conj(np.transpose(A))

    prop = sp.lil_matrix((dim**2 * 4, dim**2 * 4), dtype=complex)
    i = [[(slice(m*dim**2, (m+1)*dim**2), slice(n*dim**2, (n+1)*dim**2))
          for n in range(4)] for m in range(4)]

    prop[i[0][0]] += -1.j * commutator(h_sys)
    prop[i[0][2]] += -multiply_raveled(adj(L), 'l') + multiply_raveled(adj(L), 'r')
    prop[i[0][1]] += multiply_raveled(L, 'l') - multiply_raveled(L, 'r')

    prop[i[1][1]] += -1.j * commutator(h_sys) - np.conj(w) * np.identity(dim**2)
    prop[i[1][0]] += np.conj(g) * multiply_raveled(adj(L), 'r')
    prop[i[1][3]] += -multiply_raveled(adj(L), 'l') - multiply_raveled(adj(L), 'r')

    prop[i[2][2]] += -1.j * commutator(h_sys) - w * np.identity(dim**2)
    prop[i[2][0]] += g * multiply_raveled(L, 'l')
    prop[i[2][3]] += -multiply_raveled(L, 'l') - multiply_raveled(L, 'r')

    prop[i[3][3]] += -1.j * commutator(h_sys) - (w + np.conj(w)) * np.identity(dim**2)
    prop[i[3][1]] += g * multiply_raveled(L, 'l')
    prop[i[3][2]] += np.conj(g) * multiply_raveled(adj(L), 'r')

    print('CORRECT')
    print(prop)

    y0 = np.zeros((4, dim, dim), dtype=complex)
    y0[0] = rho0
    rho = np.empty((len(timesteps), dim, dim), dtype=complex)
    rho[0] = rho0

    r = complex_ode(lambda t, y: prop.dot(y))\
        .set_integrator('vode', atol=1e-10, rtol=1e-10, nsteps=100) \
        .set_initial_value(y0.ravel())
    for i, t in enumerate(timesteps[1:]):
        r.integrate(t)
        rho[i + 1] = r.y.reshape((4, dim, dim))[0]

    return timesteps, rho


def simple_fermionic_meq(g, gamma, omega, h_sys):
    """@todo: Docstring for simple_fermionic_meq.

    :param g: @todo
    :param gamma: @todo
    :param omega: @todo
    :param h_sys: @todo
    :returns: @todo

    """
    dim = np.shape(h_sys)[0]
    bath = {'g': [g for _ in range(dim)],
            'gamma': [gamma for _ in range(dim)],
            'Omega': [omega for _ in range(dim)]}

    couplops = np.zeros((dim, dim, dim), dtype=complex_t)
    for i in range(dim):
        couplops[i, i, i] = 1.

    return FermionicIntegrator(bath, h_sys, couplops)
