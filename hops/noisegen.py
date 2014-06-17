"""
Functions and classes related to generating colored noise

ExponentialNoiseGenerator contains a generator for noise with correlation
function

         alpha(t) = sum_j  g_j * exp(-gamma_j * |t| - ii * Omega_j * t).

In order to obtain good results, two conditions have to be met:
    - alpha(0) should be real, otherwise the resulting correlation function
    is distorted around t=0 with larger real part and zero imaginary part
    - the propagation time should be enough for alpha to be decayed to
    approximately zero.

Usage: (see generate_statistics)

The Noise Generator is based on the Fourier-Filter algorithm described in
details in Garcia-Ojalvo, Sancho: Noise in Spacially Extended Systems.
"""
# encoding: utf-8

from __future__ import division, print_function
import numpy as np


class ExponentialNoiseGenerator(object):

    """Docstring for ExponentialNoiseGenerator. """

    def __init__(self, dt, t_steps, g, gamma, omega):
        """@todo: to be defined1.

        dt -- TODO
        t_steps -- TODO
        g -- TODO
        gamma -- TODO
        omega -- TODO

        """
        exp_term = lambda g, gamma, omega, t: \
                g * np.exp(-gamma * np.abs(t) - 1.j * omega * t)
        alpha = np.empty(2*t_steps, dtype=np.complex128)
        t = dt * np.arange(t_steps+1)
        alpha[:t_steps + 1] = np.sum(exp_term(g[None, :], gamma[None, :],
                                              omega[None, :], t[:, None]),
                                     axis=1)
        alpha[t_steps:] = np.conj(alpha[t_steps:0:-1])

        self._sqrt_j = np.sqrt(np.fft.fft(alpha))
        self._t_steps = t_steps

    def get_realization(self):
        """@todo: Docstring for get_realization.
        Returns:

        """
        eta = np.sqrt(-np.log(np.random.rand(2 * self._t_steps))) * \
                np.exp(2.j * np.pi * np.random.rand(2 * self._t_steps))
        return np.fft.ifft(np.fft.fft(eta) * self._sqrt_j)[:self._t_steps]


def generate_statistics(dt, t_steps, g, gamma, omega, realizations):
    """@todo: Docstring for generate_statistics.

    dt -- TODO
    t_steps -- TODO
    g -- TODO
    gamma -- TODO
    omega -- TODO
    Returns:

    """
    from progressbar import Monitor
    noisegen = ExponentialNoiseGenerator(dt, t_steps, g, gamma, omega)

    EZ = np.zeros(t_steps, dtype=np.complex128)
    EZZ = np.zeros(t_steps, dtype=np.complex128)
    EZccZ = np.zeros(t_steps, dtype=np.complex128)

    for i in Monitor(xrange(realizations)):
        Z = noisegen.get_realization()
        EZ += Z
        EZZ += Z * Z[0]
        EZccZ += Z * np.conj(Z[0])

    EZ /= realizations
    EZZ /= realizations
    EZccZ /= realizations

    alpha = lambda t: np.sum(g[None, :] * np.exp(-gamma[None, :] * np.abs(t[:, None])
                 - 1.j * omega[None, :] * t[:, None]), axis=1)
    t = np.arange(t_steps * dt)

    return EZ, EZZ, EZccZ, alpha(t)
