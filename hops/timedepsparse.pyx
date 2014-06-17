from __future__ import division, print_function

import numpy as np
cimport numpy as np
from numpy import int, complex128
from numpy cimport int_t, float64_t as float_t, complex128_t as complex_t
from collections import namedtuple
from libcpp.vector cimport vector
from libcpp cimport bool


cdef extern from "sparsetools.hpp":
    cdef void coo_tocsr[I, T](I n_row, I n_col, I nnz, I Ai[], I Aj[], T Ax[],
                        I Bp[], I Bj[], T Bx[])

TimeDepCSR = namedtuple('TimeDepCSR', ('indptr', 'indices', 'data', 'coeff'))

cdef class TimeDepCOO(object):

    cdef vector[int_t] _i_coo
    cdef vector[int_t] _j_coo
    cdef vector[int_t] _a_coo
    cdef vector[complex_t] _c_coo
    cdef tuple _shape

    def __init__(self, shape):
        """@todo: to be defined1.

        shape -- TODO

        """
        self._shape = shape

    def append(self, i, j, a, c=1.0):
        """@todo: Docstring for append.

        i -- TODO
        j -- TODO
        a -- TODO
        c -- TODO
        """
        self._i_coo.push_back(i)
        self._j_coo.push_back(j)
        self._a_coo.push_back(a)
        self._c_coo.push_back(c)

    def to_csr(self):
        """Converts it into a TimeDepCSR Matrix.

        Returns:
        TimeDepCSR
        """
        cdef int_t nnz = self._a_coo.size()
        cdef np.ndarray[int_t] i_csr = np.empty(self._shape[0] + 1, dtype=int)
        cdef np.ndarray[int_t] j_csr = np.empty(nnz, dtype=int)
        cdef np.ndarray[int_t] a_csr = np.empty(nnz, dtype=int)
        cdef np.ndarray[complex_t] c_csr = np.empty(nnz, dtype=complex128)

        coo_tocsr[int_t, int_t](self._shape[0], self._shape[1], nnz,
                                &self._i_coo[0], &self._j_coo[0],
                                &self._a_coo[0],
                                &i_csr[0], &j_csr[0], &a_csr[0])
        coo_tocsr[int_t, complex_t](self._shape[0], self._shape[1], nnz,
                                    &self._i_coo[0], &self._j_coo[0],
                                    &self._c_coo[0],
                                    &i_csr[0], &j_csr[0], &c_csr[0])

        return TimeDepCSR(i_csr, j_csr, a_csr, c_csr)
