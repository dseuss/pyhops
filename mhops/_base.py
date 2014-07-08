#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
from numpy import float64 as float_t, complex128 as complex_t, int as int_t
from scipy.integrate import ode


def multiply_raveled(A, side='l', kronf=np.kron, identityf=np.identity):
    """Computes the matrix-representation of the matrix multiplication
    rho -> A.rho (or rho -> rho.A if side == 'r').

    A -- n*n Matrix to multiply by

    Keyword arguments:
    side -- Determines if we represent left multiplication ('l') or right
            multiplication ('r') (default l)
    kronf -- Implementation of Kronecker product to use (default np.kron)
    identityf -- Implementation of identity-returning function to use
                 (default np.identity)
    Returns:
    Matrix of shape n^2 * n^2
    """
    if side == 'l':
        return kronf(A, identityf(A.shape[0]))
    elif side == 'r':
        return kronf(identityf(A.shape[0]), A.T)
    else:
        raise KeyError('Invalid multiplication side.')


def commutator(A, **kwargs):
    """Returns the matrix representation of rho -> A.rho - rho.A^*, where
    rho is considered to be flattened in row-major order, i.e.

                    ((1, 2), (3, 4)) -> (1, 2, 3, 4)

    A -- n*n matrix to be converted
    Returns:
    (n^2) * (n^2) matrix
    """
    return multiply_raveled(A, side='l', **kwargs) \
        - multiply_raveled(A, side='r', **kwargs)


class MasterIntegrator(object):

    """Docstring for MasterEqIntegrator. """

    def __init__(self, bath, h_sys, couplops):
        """@todo: Docstring for __init__.

        bath -- TODO
        h_sys -- TODO
        Returns:

        """
        self._g = np.ravel(bath['g']).astype(complex_t)
        self._gamma = np.ravel(bath['gamma']).astype(float_t)
        self._omega = np.ravel(bath['Omega']).astype(float_t)
        self._modes = self._g.size
        self._l_map = np.ravel([np.ones(len(g), dtype=int_t) * i
                                for i, g in enumerate(bath['g'])])
        self._h_sys = np.ascontiguousarray(h_sys, dtype=complex_t)
        self._l = np.asarray(couplops)

        self._nr_aux_states = None
        self._prop = None

    @property
    def nr_aux_states(self):
        return self._nr_aux_states

    @property
    def dim_aux(self):
        return self.nr_aux_states * self._h_sys.shape[0] * self._h_sys.shape[0]

    @property
    def dim_hs(self):
        return self._h_sys.shape[0]

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
