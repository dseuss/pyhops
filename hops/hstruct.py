#!/usr/bin/env python
# encoding: utf-8
"""
Module to manage the structure of HOPS. The hierarchy structure is determined
by the set of all valid vector indices combined with an enumeration of the
latter.

A hierarchy structure is characterized by the number of modes N (the dimension
of the vector index (k)) and the truncation condition. Currently, there are
three different criteria, which determine valid (k):

    - depth D of the hierarchy (triangular truncation): (k) is a valid vector
    index of the hierarchy, iff
                            k_1 + ... + k_N <= D

    - populated modes P: (k) is a valid vector index of the hierarchy, iff
    at most P components of (k) are non-zero.
    Example: P=2, N=3, D=4: valid [0,0,0], [4,0,0], [3,1,0]
                        not valid: [1,1,1] (3 non-zero components)

    - manual cutoff (c): (k) is a valid vector index of the hierarchy, iff
              for all i=1,...,N:   k_i <= c_i
"""

from __future__ import division, print_function
import numpy as np
from itertools import izip

INVALID_INDEX = -1


class HierarchyStructure(object):

    """Structure for the given number of modes and truncation criteria.

    Usage:
    H = HierarchyStructure(modes=3, depth=2)
    entries = H.entries         # Number of entries
    H.vecind[0]                 # Vector index k = (0,...,0)
    H.vecind[H.indab[0, 1]]     # Vector index k = (0,1,...,0)
    """

    @property
    def entries(self):
        return len(self.vecind)

    def __init__(self, modes, depth, pop_modes=None, cutoff=None):
        """
        modes -- Number of modes
        depth -- Depth of the hierarchy

        Keyword arguments:
        pop_modes     -- Max number of modes i with k_i != 0 (default modes)
        cutoff[modes] -- Manual cutoff array (default [depth, ..., depth])
        """
        self._modes = modes
        self._depth = depth
        self._pop_modes = modes if pop_modes is None else pop_modes
        cutoff = [depth] * modes if cutoff is None else cutoff
        self._cutoff = np.asarray([k if (k >= 0) and (k <= depth) else depth
                                   for k in cutoff])

        self.vecind = list()
        self._recursive_indices((), 0)

        self.vecind = np.asarray(self.vecind, dtype=int)
        self.indab, self.indbl = self._lookup_coupling_indices()

    def _recursive_indices(self, k, curr_pop):
        """Recursively construct the hierachy in self.vecind

        k       -- Tuple with already determined entries.
        curr_pop -- Number of currently "populated" modes.

        Usage (to construct full hierarchy): `self._recursive_indices((), 0)`
        """
        if len(k) >= self._modes - 1:
            self._add(k + (0,))
        else:
            self._recursive_indices(k + (0,), curr_pop)

        if curr_pop >= self._pop_modes:
            return

        for i in xrange(1, self._depth - sum(k) + 1):
            if len(k) >= self._modes - 1:
                self._add(k + (i,))
            else:
                self._recursive_indices(k + (i,), curr_pop + 1)

    def _add(self, k):
        """Adds the vector index `k` to the hierarchy structure if it does not
        violate the manual cutoff criterion.

        k[modes] -- Vector index to add.
        """
        if all(np.asarray(k) <= self._cutoff):
            self.vecind.append(k)

    def _lookup_coupling_indices(self):
        """Create indices for coupling above/below in the hierarchy.
        `indab[i, j]` is the integer-index corresponding to the vector index
        `vecind[i] + e_j`.
        """
        iind = dict(izip([tuple(x) for x in self.vecind],
                         range(self.vecind.shape[0])))
        e = np.identity(self.vecind.shape[1], dtype=int)
        indab = np.empty(self.vecind.shape, dtype=int)
        indbl = np.empty(self.vecind.shape, dtype=int)

        for i, k in enumerate(self.vecind):
            indab[i] = np.asarray([iind.get(tuple(kp), INVALID_INDEX)
                                   for kp in k + e], dtype=int)
            indbl[i] = np.asarray([iind.get(tuple(km), INVALID_INDEX)
                                   for km in k - e], dtype=int)

        return indab, indbl

    def __str__(self):
        rep = []
        for i, k in enumerate(self.vecind):
            rep.append('{}: {}'.format(i, k))
            for j in xrange(self._modes):
                rep.append('  +{}: {} | -{}: {}'.format(j, self.indab[i][j], j,
                                                        self.indbl[i][j]))
        return '\n'.join(rep)
