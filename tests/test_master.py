#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
from mhops.master import commutator


def test_commutator():
    dim = 10
    A = np.random.random((dim, dim)) + 1.j * np.random.random((dim, dim))
    rho = np.random.random((dim, dim)) + 1.j * np.random.random((dim, dim))

    Arho1 = np.dot(A, rho) - np.dot(rho, A)
    Arho2 = np.dot(commutator(A), np.ravel(rho))

    assert np.max(np.abs((Arho1.ravel() - Arho2))) \
        < np.finfo(np.float64).eps * dim**2
