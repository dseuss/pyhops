# encoding: utf-8

from __future__ import division, print_function
import numpy as np
cimport numpy as np
from numpy import int, complex128
from numpy cimport int_t, float64_t as float_t, complex128_t as complex_t
import scipy.sparse as sparse
from libcpp.vector cimport vector
from libcpp cimport bool
from hstruct import INVALID_INDEX


cdef int_t CINVALID_INDEX = INVALID_INDEX


cdef extern from "sparsetools.hpp":
    cdef void coo_tocsr[I, T](I n_row, I n_col, I nnz, I Ai[], I Aj[], T Ax[],
                        I Bp[], I Bj[], T Bx[])


cdef void _add_block(vector[int_t]& i_coo, vector[int_t]& j_coo,
                     vector[complex_t]& a_coo, int_t pos_i, int_t pos_j,
                     complex_t[:, :] h):
    """Add the matrix h as a block to the sparse matrix A = (cooI, cooJ, cooA).
    It assumes that A can be subdivided into blocks of the same shape as h and
    adds h to the (pos_i, pos_j)-th block, i.e. h[0,0] is added to the
    (h.shape[0] * pos_i, h.shape[1] * pos_j)-the entry.

    i_coo -- row indices of sparse matrix in COO format
    j_coo -- col indices of sparse matrix in COO format
    a_coo -- entries of sparse matrix in COO format
    pos_i -- block row index
    pos_i -- block col index
    h     -- value to add
    """
    cdef int_t i
    cdef int_t j
    cdef int_t dim_i = h.shape[0]
    cdef int_t dim_j = h.shape[1]

    for i in range(dim_i):
        for j in range(dim_j):
            i_coo.push_back(pos_i * dim_i + i)
            j_coo.push_back(pos_j * dim_j + j)
            a_coo.push_back(h[i, j])


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

    cdef vector[complex_t] a_coo
    cdef vector[int_t] i_coo
    cdef vector[int_t] j_coo

    cdef int_t iind
    cdef int_t mode
    cdef int_t couplind


    cdef np.ndarray[int_t, ndim=1] k
    cdef np.ndarray[float_t, ndim=2, mode='c'] identity = np.identity(dim)

    for iind in range(nr_aux):
        k = vecind[iind]
        _add_block(i_coo, j_coo, a_coo, iind, iind,
                   -1.j * h_sys - np.dot(k, w) * identity)

        for mode in range(nr_modes):
            couplind = indbl[iind, mode]
            if couplind != CINVALID_INDEX:
                i_coo.push_back(dim * iind + l_map[mode])
                j_coo.push_back(dim * couplind + l_map[mode])
                a_coo.push_back(k[mode] * g[mode])

            couplind = indab[iind, mode]
            if couplind != CINVALID_INDEX:
                i_coo.push_back(dim * iind + l_map[mode])
                j_coo.push_back(dim * couplind + l_map[mode])
                a_coo.push_back(-1)

            elif with_terminator:
                for iind_t, mode_t, val_t in terminator(iind, mode,
                                                       vecind, indab, indbl,
                                                       g, w, l_map):
                    i_coo.push_back(dim * iind + l_map[mode_t])
                    j_coo.push_back(dim * iind_t + l_map[mode_t])
                    a_coo.push_back(val_t)

    cdef int_t nnz = a_coo.size()
    cdef np.ndarray[int_t] i_csr = np.empty(size + 1, dtype=int)
    cdef np.ndarray[int_t] j_csr = np.empty(nnz, dtype=int)
    cdef np.ndarray[complex_t] a_csr = np.empty(nnz, dtype=complex128)

    coo_tocsr[int_t, np.complex128_t](size, size, nnz,
                                   &i_coo[0], &j_coo[0], &a_coo[0],
                                   &i_csr[0], &j_csr[0], &a_csr[0])
    return i_csr, j_csr, a_csr


cdef terminator(int_t iind,
                int_t mode,
                np.ndarray[int_t, ndim=2, mode='c'] vecind,
                np.ndarray[int_t, ndim=2, mode='c'] indab,
                np.ndarray[int_t, ndim=2, mode='c'] indbl,
                np.ndarray[complex_t] g,
                np.ndarray[complex_t] w,
                np.ndarray[int_t] l_map):
    """Returns the terminator replacement for psi[k + e_mode], where
    k = vecind[iind]:
            psi[k+e_j] = sum_i (k+e_j)_i g_i / dot(k+e_j, w)        (*)
                                   * l_map[i] psi[k+e_j-e_i]

    iind -- Integer index for the current aux state (k)
    mode -- Mode for which k is increased by one

    Returns:
    List of tuples (iind_t, mode_t, val_t), where
    iind_t -- Integer index for (k + e_j - e_i) (j=mode, i=mode_t)
    mode_t -- Mode which is subtracted from (k + e_j)
    val_t  -- Prefactor for terminator (i.e. everything in the first line
              of (*))
    """
    res = []
    cdef int_t nr_modes = vecind.shape[1]
    cdef complex_t kdw = np.dot(vecind[iind], w) + w[mode]
    cdef int_t mode_t
    cdef int_t iind_t

    for mode_t in range(nr_modes):
        if mode_t == mode:
            res.append((iind, mode, -(vecind[iind, mode] + 1) * g[mode] / kdw))
            continue

        if indbl[iind, mode_t] == CINVALID_INDEX: continue

        iind_t = indab[indbl[iind, mode_t], mode]
        if iind_t == CINVALID_INDEX: continue

        res.append((iind_t, mode_t, -vecind[iind, mode_t] * g[mode_t] / kdw))

    return res
