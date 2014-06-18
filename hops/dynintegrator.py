"""
TODO
"""
#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np

from hops.specintegrator import SpectrumHierarchyIntegrator
from hops.noisegen import ExponentialNoiseGenerator as ExpNG
from hops.libhint import hint
from hops.timedepsparse import TimeDepCOO


class LinearHierarchyIntegrator(SpectrumHierarchyIntegrator):

    """Docstring for LinearHierarchyIntegrator. """

    def __init__(self, t_length, t_steps, struct, bath, h_sys, psi0=None,
                 with_terminator=True, seed=0):
        """See also SpectrumHierarchyIntegrator.__init__

        t_length -- full propagation time
        t_steps  -- number of time steps
        """
        np.random.seed(seed)
        dt = t_length / (t_steps - 1)
        dim = h_sys.shape[0]

        SpectrumHierarchyIntegrator.__init__(self, struct, bath, h_sys,
                                             with_terminator)
        self._t_length = t_length
        self._t_steps = t_steps
        self._psi0 = np.asarray([1] + [0] * (dim - 1), dtype=np.complex128) \
                if psi0 is None else psi0

        self._noisegen = [ExpNG(dt/2, 2*t_steps, self._g[s], self._gamma[s],
                                self._omega[s])
                          for s in [self._l_map == i for i in range(dim)]]

        self._noiseprop = self._setup_noise_propagator()

    def _setup_noise_propagator(self):
        """@todo: Docstring for _setup_noise_propagator.

        struct -- TODO
        Returns:

        """
        nprop = TimeDepCOO((self.nr_aux_states, self.nr_aux_states))
        dim = self._h_sys.shape[0]
        dim_iterator = range(dim)

        for iind in xrange(self.nr_aux_states):
            for i in dim_iterator:
                nprop.append(iind * dim + i, iind * dim + i, i)

        return nprop.to_csr()

    def get_trajectory(self, noise=None):
        """@todo: Docstring for get_trajectory.

        noise -- TODO
        Returns:

        """
        if noise is None:
            noise = np.asarray([ng.get_realization() for ng in self._noisegen],
                               dtype=np.complex128)

        psi = hint.calc_trajectory_lin(t_length=self._t_length,
                                       t_steps=self._t_steps,
                                       dim_hs=self._h_sys.shape[0],
                                       nr_aux_states=self.nr_aux_states,
                                       nr_noise=self._h_sys.shape[0],
                                       psi0=self._psi0,

                                       lin_i=self._prop.indptr,
                                       lin_j=self._prop.indices,
                                       lin_a=self._prop.data,

                                       noise_i=self._noiseprop.indptr,
                                       noise_j=self._noiseprop.indices,
                                       noise_a=self._noiseprop.data,
                                       noise_c=self._noiseprop.coeff,
                                       noise=noise)

        return np.transpose(psi)


class NonlinHierarchyIntegrator(LinearHierarchyIntegrator):

    """Docstring for NonlinHierarchyIntegrator. """

    def __init__(self, t_length, t_steps, struct, bath, h_sys, psi0=None,
                 with_terminator=True):
        """See LinearHierarchyIntegrator.__init__"""
        LinearHierarchyIntegrator.__init__(self, t_length, t_steps, struct,
                                           bath, h_sys, psi0, with_terminator)

