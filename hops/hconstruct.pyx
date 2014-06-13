# encoding: utf-8

from __future__ import division, print_function
import numpy as np
cimport numpy as np
from numpy import int, complex128
from numpy cimport int_t, complex128_t as complex_t
import scipy.sparse as sparse
from libcpp.vector cimport vector
from libcpp cimport bool
from hstruct import INVALID_INDEX

cdef extern from "sparsetools.hpp":
    cdef void coo_tocsr[I, T](I n_row, I n_col, I nnz, I Ai[], I Aj[], T Ax[],
                        I Bp[], I Bj[], T Bx[])



cdef void _add_block(vector[int_t]& cooI, vector[int_t]& cooJ,
                     vector[complex_t]& cooA, int_t pos_i, int_t pos_j,
                     complex_t[:, :] h):
    cdef int_t i
    cdef int_t j
    cdef int_t dim_i = h.shape[0]
    cdef int_t dim_j = h.shape[1]

    for i in range(dim_i):
        for j in range(dim_j):
            cooI.push_back(pos_i * dim_i + i)
            cooJ.push_back(pos_j * dim_j + j)
            cooA.push_back(h[i, j])


def setup_linear_propagator(np.ndarray[int_t, ndim=2, mode='c'] vecind,
                            np.ndarray[int_t, ndim=2, mode='c'] indab,
                            np.ndarray[int_t, ndim=2, mode='c'] indbl,
                            np.ndarray[complex_t, ndim=2, mode='c'] h_sys,
                            np.ndarray[complex_t] g,
                            np.ndarray[complex_t] w,
                            np.ndarray[int_t] l_map,
                            bool with_terminator):
    """@todo: Docstring for _setup_linear_propagator.

    """
    cdef int_t dim = h_sys.shape[0]
    cdef int_t nr_aux = vecind.shape[0]
    cdef int_t nr_modes = vecind.shape[1]
    cdef int_t size = dim * nr_aux

    cdef vector[complex_t] cooA
    cdef vector[int_t] cooI
    cdef vector[int_t] cooJ

    cdef int_t iind
    cdef int_t mode
    cdef int_t couplind

    cdef int_t CINVALID_INDEX = INVALID_INDEX

    cdef np.ndarray[np.int_t, ndim=1] k

    for iind in range(nr_aux):
        k = vecind[iind]
        _add_block(cooI, cooJ, cooA, iind, iind,
                   -1.j * h_sys - np.dot(k, w) * np.identity(dim))

        for mode in range(nr_modes):
            couplind = indbl[iind, mode]
            if couplind != CINVALID_INDEX:
                cooI.push_back(dim * iind + l_map[mode])
                cooJ.push_back(dim * couplind + l_map[mode])
                cooA.push_back(k[mode] * g[mode])

            couplind = indab[iind, mode]
            if couplind != CINVALID_INDEX:
                cooI.push_back(dim * iind + l_map[mode])
                cooJ.push_back(dim * couplind + l_map[mode])
                cooA.push_back(-1)

            # elif with_terminator:
            #     for ent_t, mode_t, val_t in self._terminator(ent, mode):
            #         prop[dim * ent + self._l_map[mode_t],
            #              dim * ent_t + self._l_map[mode_t]] += val_t

    cdef int_t nnz = cooA.size()
    cdef np.ndarray[int_t] csrI = np.empty(size + 1, dtype=int)
    cdef np.ndarray[int_t] csrJ = np.empty(nnz, dtype=int)
    cdef np.ndarray[complex_t] csrA = np.empty(nnz, dtype=complex128)

    coo_tocsr[int_t, np.complex128_t](size, size, nnz,
                                   &cooI[0], &cooJ[0], &cooA[0],
                                   &csrI[0], &csrJ[0], &csrA[0])
    return csrI, csrJ, csrA
