#!/usr/bin/env python
# encoding: utf-8
"""
Simulation of an arbitrary system coupled to a finite fermionic environment,
such that the system operators __commute__ with the bath operators.
The full Hamiltonian reads

    H = H_sys  +  Σ_j ω_j c†_j c_j  +  Σ_j (g^*_j L c†_j + g_j L† c_j)

with fermionic ladder operators {d_i, d†_j} = δ_ij. Here, we assume that
system and bath are distinguishable. For details see [1, IV].

[1] Shi, Zhao, Yu: Non-Markovian Fermionic Stochastic Schroedinger Equation for
    Open System Dynamics; arXiv:1203.2219 [quant-ph]
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as pl
import scipy.sparse as sp
from itertools import izip

import physics.qustat as qs
import mhops.fermionic as fm
from tools.sci import zodeint


def full_hamiltonian(h_sys, coupl, omega, strength):
    """Returns the hamiltonian for a system interacting with a finite fermionic
    environment. Note that the convention kron(system, environment) is used.

    :param h_sys: System hamiltonian
    :param coupl: Coupling operator L
    :param omega: Eigenfrequencies of the environmental fermions w_j
    :param strength: Coupling strenghts to the environmental modes g_j
    :returns: Sparse matrix representation of the Hamiltoninan H

    """
    bathop = qs.annhilation_operators(len(omega))
    dim_sys, dim_env = np.shape(h_sys)[0], bathop[0].shape[0]
    I = sp.identity
    adj = lambda A: np.conj(np.transpose(A))

    hamil = sp.kron(h_sys, I(dim_env)) \
            + sp.kron(I(dim_sys), sum([w * d.T * d for w, d in izip(omega, bathop)])) \
            + sum([np.conj(g) * sp.kron(coupl, adj(c)) + g * sp.kron(adj(coupl), c)
                   for g, c in izip(strength, bathop)])

    hamil = hamil.tocsr()
    hamil.eliminate_zeros()
    return hamil


def full_state(psi0, nr_fermions):
    """Returns the full system-environment state |psi_0> \otimes |vac>.
    Note that the convention kron(system, environment) is used.

    :param psi0: @todo
    :returns: @todo

    """
    vac = qs.tensor([np.array([1, 0])] * nr_fermions)
    vac /= np.sqrt(np.vdot(vac, vac))
    return np.kron(psi0, vac)


def reduced_rho(psi, dim_sys):
    """Returns the reduced density operator from the full system-environment
    pure state psi. Note that the conventions kron(system, environment) is
    used.

    :param psi: @todo
    :returns: @todo

    """
    dim_env = psi.size // dim_sys
    psi = psi.reshape((dim_sys, dim_env))
    rho = np.conj(psi)[:, :, None, None] * psi[None, None, :, :]
    rho_red = np.trace(rho, axis1=1, axis2=3)
    return rho_red


def analytic_rho12(t, psi0, omega, strength):
    """Computes the analytic solution of the dissipative quantum dot coupled
    to one fermion in resonance (w_sys = w_1)

    :param t: Time steps where the analytic solution should be evaluated
    :param psi0: initial pure state of the system
    :param omega: Frequency of the system (and the oscillator)
    :param strengh: Coupling strength of the env. to the fermion
    :returns: Array of length `len(t)` with matrix element rho_12(t) of the
              system's reduced density operator
    """
    return reduced_rho(psi0, 2)[0, 1] * np.exp(1.j * omega * t) \
        * np.cos(strength*t)


if __name__ == '__main__':
    h_sys = .5 * 4 * np.diag((1, -1))
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_m = np.array([[0, 0], [1, 0]])
    g = [1., 1., 2., 4.]
    omega = [2., 1., 4., -1.5]
    cl = ['r', 'g']
    t, dt = np.linspace(0, 10, 1000, retstep=True)
    psi_0_sys = np.asarray([.6, .8])

    def plot(t, rho, **kwargs):
        for i in range(2):
            pl.plot(t, np.real(rho[:, i, i]), color=cl[i], **kwargs)
        # pl.plot(t, np.real(rho[:, 0, 1]), color=cl[0], **kwargs)
        # pl.plot(t, np.imag(rho[:, 0, 1]), color=cl[1], **kwargs)

    # Analytic solution #######################################################
    # pl.plot(t, np.real(analytic_rho12(t, psi_0_sys, omega[0], g[0])),
    #         lw=2, color='k')
    # pl.plot(t, np.imag(analytic_rho12(t, psi_0_sys, omega[0], g[0])),
    #         lw=2, color='k')

    # Full Hilbert space solution #############################################
    prop = -1.j * full_hamiltonian(h_sys, sigma_m, omega, np.sqrt(g))
    psi0 = full_state(psi_0_sys, len(g)).astype(complex)
    _, psi = zodeint(lambda t, y: prop.dot(y), psi0, t,
                     rtol=10e-10, atol=10e-10)
    rho = np.asarray([reduced_rho(psi_t, 2) for psi_t in psi])
    print(np.max(np.abs(np.trace(rho.T) - 1)))
    rho /= np.trace(rho.T)[:, None, None]
    plot(t, rho, ls='--')

    # Manual hierarchy test ###################################################
    # _, rho = fm._testcase_one_mode(h_sys,
    #                                np.conj(psi_0_sys)[:, None] * psi_0_sys[None, :],
    #                                g[0],
    #                                1.j * omega[0],
    #                                sigma_m,
    #                                t)

    # rho /= np.trace(rho.T)[:, None, None]
    # plot(t, np.conj(rho), ls=':')

    # General hierarchy #######################################################
    bath = {'g': [g], 'gamma': [[0.] * len(g)], 'Omega': [omega]}
    meq = fm.FermionicIntegrator(bath, h_sys, couplops=[sigma_m])
    t, rho = meq.get_rho(dt, t[-1], psi0=np.array(psi_0_sys))
    print(np.max(np.abs(np.trace(rho.T) - 1)))
    rho /= np.trace(rho.T)[:, None, None]
    plot(t, rho, ls=':')

    # # meq = bm.BosonicIntegrator(4, bath, .5 * sigma_z, couplops=[sigma_m])
    # meq = fm.FermionicIntegrator(bath, .5 * sigma_z, couplops=[sigma_m])
    # t, rho = meq.get_rho(dt, t[-1], psi0=np.array(psi_0_sys))
    # rho /= trace(rho)[:, None, None]
    # plot(t, rho, ls=':')

    pl.show()
