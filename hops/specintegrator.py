#!/usr/bin/env python
# encoding: utf-8
"""
Main module for the spectrum-hierarchy-integrator. Contains the necessary
setup as well as the integration routines. For all that follows we assume a
decomposition of the bath correlation function alpha(t):

    alpha_n(t) = sum_j  g^n_j * exp(-gamma^n_j * |t| - ii * Omega^n_j * t).

The process corresponding to alpha_n has coupling operator L_n. Currently,
only the projectors L_n = |pi_n><pi_n| (n = 1, ..., dim_system) are
implemented.
The parameters of the bcf are passed as flat (1D) arrays gT, gammaT, OmegaT
(the T stands for tilde and is dropped in the true code). The connect the
entries of the flat arrays to the modes of individual L_n, we use the array
Lmap. The parameters gT[m], gammaT[m], and OmegaT[m] belong to the modes
coupling with coupling operator L_Lmap[m].

The spectrum-hierarchy integrates the linear NMSSE hierarchy with no noise

    d_t psi^(k) = (-ii*H_sys - k.w) psi^(k)
                + sum_j k_j g_j L_Lmap[j] psi^(k-e_j)
                - sum_j L^*_Lmap[j] psi^(k+e_j).
"""

from __future__ import division, print_function

import numpy as np
from numpy import complex128 as complex_t, float64 as float_t
from scipy.sparse import csr_matrix as CSRMatrix, lil_matrix as LILMatrix
from scipy.integrate import ode

from hops.hstruct import HierarchyStructure, INVALID_INDEX
from hops.libhint import hint
import hops.hconstruct as hcons
import hops.specfun as specfun


class SpectrumHierarchyIntegrator(object):

    """Integrator for the linear hierarchy without noise.

    Usage: (see simple_spectrum_hierarchy)
    """

    def __init__(self, struct, bath, h_sys, with_terminator=True):
        """
        struct -- HierarchyStructure to use
        bath   -- Dict with keys 'g', 'gamma', 'Omega' containing the
                  parameters for the bath correlation functions. Each entry is
                  a list of lists, where i.e. g[i][j] is the coupling strength
                  for the j-th mode of the i-th coupling operator.

                  Alternatively, pass an instance of OscillatorBath.
        h_sys  -- @todo

        Keyword arguments:
        with_terminator -- Use terminator for truncated aux. states
        """
        self._g = np.ravel(bath.get('g')).astype(complex_t)
        self._gamma = np.ravel(bath.get('gamma')).astype(float_t)
        self._omega = np.ravel(bath.get('Omega')).astype(float_t)
        self._l_map = np.ravel([np.ones(len(g)) * i
                                for i, g in enumerate(bath['g'])]).astype(int)
        self._h_sys = np.asarray(h_sys, dtype=complex_t)
        self._with_terminator = with_terminator

        self._prop = self._setup_linear_propagator(struct)

    @property
    def nr_equations(self):
        return self._prop.shape[0]

    def _setup_linear_propagator(self, struct):
        """Sets up the noise-independent, linear propagator

        Returns:
        The propagator as CSR Matrix
        """
        i, j, a = hcons.setup_linear_propagator(struct.vecind,
                                                struct.indab,
                                                struct.indbl,
                                                self._h_sys,
                                                self._g,
                                                self._gamma + 1.j*self._omega,
                                                self._l_map,
                                                self._with_terminator)
        csr = CSRMatrix((a, j, i))
        print('Vorher: {}'.format(csr.nnz))
        csr.sum_duplicates()
        csr.eliminate_zeros()
        print('Nachher: {}'.format(csr.nnz))
        return csr

    def get_trajectory(self, t_length, t_steps, psi0=None):
        """Calculates the solution of the linear hierarchy without noise

        t_length -- full propagation time
        t_steps  -- number of time steps

        Keyword arguments:
        psi0[dim] -- initial state of the system (default [1,1,...]/sqrt(dim))
                     (alternatively pass array of size self.nr_equations and
                      dype complex_t, which is used as computing state. After
                      propagation it contains the full hierarchy state, which
                      can be used to continue propagation.)
        extern    -- use compiled integrator (default True)

        Returns:
        psi[t_steps, dim] -- Trajectory for Z_t = 0
        """
        dim = self._h_sys.shape[0]
        if psi0 is None:
            psi0 = np.ones(dim, dtype=complex_t) / np.sqrt(dim)
        if psi0.size == dim:
            psi0 = np.append(psi0, np.zeros(self.nr_equations - dim,
                                            dtype=complex_t))
        elif psi0.size != self.nr_equations:
            raise RuntimeError("specintegrator.py:get_trajectory: psi0 has wrong shape")

        psi = hint.calc_trajectory_lin(t_length, t_steps, dim, psi0,
                                       self._prop.indptr,
                                       self._prop.indices,
                                       self._prop.data)
        return np.transpose(psi)

    def get_spectrum(self, dw, t_steps, psi0=None, wmin=None, wmax=None,
                     sigma_w=0):
        """Calculates the spectrum using the linear hierarchy without noise

        dw      -- Resolution of the spectrum in frequency space
        t_steps -- Number of time steps used in propagation

        Keyword arguments:
        psi0[dim] -- initial state of the system (default [1,1,...]/sqrt(dim))
        extern    -- use compiled integrator (default True)
        wmin      -- lower bound for the frequency domain of the spectrum
                     (default -oo)
        wmax      -- upper bound for the frequency domain of the spectrum
                     (default +oo)
        sigma_w   -- width in frequency space of Gaussian convoluted with the
                     spectrum (default 0)

        Returns:
        w[:] -- Frequencies of the spectrum
        A[:] -- Spectrum A(w)
        """
        t_length = 2 * np.pi / dw
        psi = self.get_trajectory(t_length, t_steps, psi0)
        dt = t_length / (t_steps - 1)
        corr = np.sum(psi * np.conj(psi[0]), axis=1)
        w, A = specfun.fourier(corr, dt, output_w=True, hermitian=True,
                               sigma_w=sigma_w)

        ## Cut out values which are out of bound
        sel = np.ones(w.size, dtype=bool)
        if wmin is not None:
            sel *= w >= wmin
        if wmax is not None:
            sel *= w <= wmax

        return w[sel], np.real(A[sel])


def simple_spectrum_hierarchy(depth, g, gamma, omega, h_sys, **kwargs):
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
    struct = HierarchyStructure(len(g) * dim, depth)
    bath = {'g': [g for _ in range(dim)],
            'gamma': [gamma for _ in range(dim)],
            'Omega': [omega for _ in range(dim)]}

    return SpectrumHierarchyIntegrator(struct, bath, h_sys, **kwargs)

if __name__ == '__main__':
    from timer import Timer
    hier = simple_spectrum_hierarchy(1, [1, 1], [2, 2], [3, 3],
                                     np.zeros([2, 2]))
    print(hier._struct.entries)
    # with Timer('Intern'):
    #     hier.get_trajectory(100, 20000, extern=False)
    with Timer('Extern'):
        hier.get_trajectory(100, 20000, extern=True)
