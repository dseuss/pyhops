#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
from numpy import complex128 as complex_t, float64 as float_t, int as int_t
import scipy.sparse as sparse
from scipy.integrate import ode

from hops.hstruct import HierarchyStructure, INVALID_INDEX


def multiply_raveled(A, side='l'):
    """Computes the matrix-representation of the matrix multiplication
    rho -> A.rho (or rho -> rho.A if side == 'r').

    A -- n*n Matrix to multiply by

    Keyword arguments:
    side -- Determines if we represent left multiplication ('l') or right
            multiplication ('r') (default l)
    Returns:
    Matrix of shape n^2 * n^2
    """
    if side == 'l':
        return np.kron(A, np.identity(A.shape[0]))
    elif side == 'r':
        return np.kron(np.identity(A.shape[0]), A.T)
    else:
        raise KeyError('Invalid multiplication side.')


def commutator(A):
    """Returns the matrix representation of rho -> A.rho - rho.A^*, where
    rho is considered to be flattened in row-major order, i.e.

                    ((1, 2), (3, 4)) -> (1, 2, 3, 4)

    A -- n*n matrix to be converted
    Returns:
    (n^2) * (n^2) matrix
    """
    return multiply_raveled(A, side='l') - multiply_raveled(A, side='r')


class MasterEqIntegrator(object):

    """Docstring for MasterEqIntegrator. """

    def __init__(self, depth, bath, h_sys):
        """@todo: Docstring for __init__.

        depth -- TODO
        modes -- TODO
        bath -- TODO
        h_sys -- TODO
        Returns:

        """
        self._g = np.ravel(bath.get('g')).astype(complex_t)
        self._gamma = np.ravel(bath.get('gamma')).astype(float_t)
        self._omega = np.ravel(bath.get('Omega')).astype(float_t)
        self._modes = self._g.size
        self._l_map = np.ravel([np.ones(len(g), dtype=int_t) * i
                                for i, g in enumerate(bath['g'])])
        self._h_sys = np.ascontiguousarray(h_sys, dtype=complex_t)
        self._l = np.zeros((self.dim_hs, ) * 3)
        for i in range(self.dim_hs):
            self._l[i, i, i] = 1.

        struct = HierarchyStructure(2 * self._modes, depth)
        self._nr_aux_states = struct.entries
        self._prop = self._setup_propagator(struct)

    @property
    def nr_aux_states(self):
        return self._nr_aux_states

    @property
    def dim_aux(self):
        return self.nr_aux_states * self._h_sys.shape[0] * self._h_sys.shape[0]

    @property
    def dim_hs(self):
        return self._h_sys.shape[0]

    def _setup_propagator(self, struct):
        """@todo: Docstring for _setup_propagator.

        struct -- TODO
        Returns:

        """
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

    def get_rho(self, dt, t_length, psi0=None):
        """@todo: Docstring for get_rho.

        t_length -- TODO
        t_steps -- TODO
        psi0 -- TODO
        Returns:

        """
        if psi0 is None:
            psi0 = np.zeros(self.dim_hs, dtype=complex_t)
            psi0[0] = 1.

        rho = np.zeros(self.dim_aux, dtype=complex_t)
        rho[:self.dim_hs**2] = np.ravel(np.conj(psi0[:, None]) * psi0[None, :])

        integ = ode(lambda t, x: self._prop.dot(x)) \
            .set_integrator('zvode', with_jacobian=False) \
            .set_initial_value(rho)
        ode.set_initial_value

        times = [0.0]
        rho_t = [rho[:self.dim_hs**2]]
        while integ.successful() and integ.t < t_length:
            integ.integrate(integ.t + dt)
            times += [integ.t]
            rho_t += [integ.y[:self.dim_hs**2]]

        return np.asarray(times), \
            np.asarray(rho_t).reshape((len(times), self.dim_hs, self.dim_hs))


def simple_master_hierarchy(depth, g, gamma, omega, h_sys):
    """Spectrum hierarchy with identical, independent baths with bcf
            alpha_i(t) = sum_j g_j * exp(-gamma_j|t| - ii * Omega_j * t)

    depth          -- depth of the hierarchy
    g[modes]       -- coupling strengths of the modes
    gamma[modes]   -- damping sof the modes
    omega[modes]   -- central frequencies of the modes
    h_sys[dim,dim] -- system hamiltonian

    Usage:
    hier = simple_spectrum_hierarchy(4, [1], [2], [3], [[1, 2], [2, 1]])
    psi = hier.get_trajectory(100, 20000)

    """
    dim = np.shape(h_sys)[0]
    bath = {'g': [g for _ in range(dim)],
            'gamma': [gamma for _ in range(dim)],
            'Omega': [omega for _ in range(dim)]}

    return MasterEqIntegrator(depth, bath, h_sys)
