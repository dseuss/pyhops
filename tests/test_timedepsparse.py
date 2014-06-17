from hops.timedepsparse import TimeDepCOO


def test_creation():
    A = TimeDepCOO((2, 2))
    A.append(0, 0, 0, 0.5)
    A.append(0, 1, 0, 1.0)
    A.append(1, 0, 1, 1.5)

    A = A.to_csr()
    assert all(A.indptr == [0, 2, 3])
    assert all(A.indices == [0, 1, 0])
    assert all(A.data == [0, 0, 1])
    assert all(A.coeff == [0.5, 1., 1.5])


def test_duplicates():
    A = TimeDepCOO((2, 2))
    A.append(0, 0, 0, 0.5)
    A.append(0, 1, 0, 1.0)
    A.append(0, 1, 0, 1.0)
    A.append(1, 0, 1, 1.5)

    A = A.to_csr()
    assert all(A.indptr == [0, 3, 4])
    assert all(A.indices == [0, 1, 1, 0])
    assert all(A.data == [0, 0, 0, 1])
    assert all(A.coeff == [0.5, 1., 1., 1.5])
