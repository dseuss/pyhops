#!/usr/bin/env python
# encoding: utf-8
"""
Simulation of an arbitrary system coupled to a finite fermionic environment,
such that the system operators __commute__ with the bath operators.
The full Hamiltonian reads

    H = H_sys  +  Σ_j ω_j c†_j c_j  +  Σ_j (g^*_j L c†_j + g_j L† c_j)

with fermionic ladder operators {d_i, d†_j} = δ_ij. Here, we assume that
system and bath are distinguishable. For details see [1, IV].

[1] Shi, Zhao, Yu: Non-Markovian Fermionic Stochastic Schrödinger Equation for
    Open System Dynamics; arXiv:1203.2219 [quant-ph]
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as pl
import scipy.sparse as sp
from itertools import izip

import physics.qustat as qs
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
    return rho_red / np.trace(rho_red)


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
    sigma_z = np.diag((1, -1))
    sigma_m = np.array([[0, 0], [1, 0]])
    prop = -1.j * full_hamiltonian(.5 * sigma_z, sigma_m, [1.], [10.])
    t, dt = np.linspace(0, 10, 1000, retstep=True)
    psi_0_sys = [.6, .8]
    psi0 = full_state(psi_0_sys, 1).astype(complex)
    t, psi = zodeint(lambda t, y: prop.dot(y), psi0, t,
                     rtol=10e-10, atol=10e-10)

    rho = np.asarray([reduced_rho(psi_t, 2) for psi_t in psi])
    print(rho.shape)

    pl.plot(t, np.real(rho[:, 0, 0]))
    pl.plot(t, np.real(rho[:, 1, 1]))
    pl.show()
