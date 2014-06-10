from hops.hstruct import HierarchyStructure, INVALID_INDEX


def test_completeness():
    H = HierarchyStructure(2, 3)
    vecind = [tuple(k) for k in H.vecind]

    assert (0, 0) in vecind

    assert (1, 0) in vecind
    assert (0, 1) in vecind

    assert (2, 0) in vecind
    assert (1, 1) in vecind
    assert (0, 2) in vecind

    assert (3, 0) in vecind
    assert (2, 1) in vecind
    assert (1, 2) in vecind
    assert (0, 3) in vecind

    assert (1, 3) not in vecind
    assert (0, 8) not in vecind
    assert (4, 3) not in vecind

    assert H.entries == 10


def test_limited():
    H = HierarchyStructure(3, 3, pop_modes=2)
    vecind = [tuple(k) for k in H.vecind]

    assert (0, 0, 0) in vecind

    assert (0, 0, 1) in vecind
    assert (0, 0, 2) in vecind
    assert (0, 0, 3) in vecind
    assert (0, 1, 0) in vecind
    assert (0, 2, 0) in vecind
    assert (0, 3, 0) in vecind
    assert (1, 0, 0) in vecind
    assert (2, 0, 0) in vecind
    assert (3, 0, 0) in vecind

    assert (0, 1, 1) in vecind
    assert (0, 1, 2) in vecind
    assert (0, 2, 1) in vecind

    assert (1, 0, 1) in vecind
    assert (1, 0, 2) in vecind
    assert (2, 0, 1) in vecind

    assert (1, 1, 0) in vecind
    assert (1, 2, 0) in vecind
    assert (2, 1, 0) in vecind

    assert (1, 1, 1) not in vecind
    assert (2, 1, 1) not in vecind
    assert (0, 1, 4) not in vecind

    assert H.entries == 19


def test_cutoff():
    H = HierarchyStructure(3, 2, cutoff=[-1, 1, 0])
    vecind = [tuple(k) for k in H.vecind]
    print(vecind)

    assert (0, 0, 0) in vecind

    assert (0, 1, 0) in vecind

    assert (1, 0, 0) in vecind
    assert (1, 1, 0) in vecind
    assert (2, 0, 0) in vecind

    assert (3, 0, 0) not in vecind
    assert (0, 0, 1) not in vecind
    assert (1, 1, 1) not in vecind
    assert (2, 1, 1) not in vecind
    assert (0, 1, 4) not in vecind
    assert (1, 0, 2) not in vecind

    assert H.entries == 5


def test_cutoff_limited():
    H = HierarchyStructure(3, 3, pop_modes=2, cutoff=[-1, -1, 1])
    vecind = [tuple(k) for k in H.vecind]

    assert (0, 0, 0) in vecind

    assert (0, 0, 1) in vecind
    assert (0, 1, 0) in vecind
    assert (0, 2, 0) in vecind
    assert (0, 3, 0) in vecind
    assert (1, 0, 0) in vecind
    assert (2, 0, 0) in vecind
    assert (3, 0, 0) in vecind

    assert (0, 1, 1) in vecind
    assert (0, 2, 1) in vecind

    assert (1, 0, 1) in vecind
    assert (2, 0, 1) in vecind

    assert (1, 1, 0) in vecind
    assert (1, 2, 0) in vecind
    assert (2, 1, 0) in vecind

    assert (1, 1, 1) not in vecind
    assert (2, 1, 1) not in vecind
    assert (0, 1, 4) not in vecind
    assert (1, 0, 2) not in vecind
    assert (1, 1, 2) not in vecind

    assert H.entries == 15


def test_coupling():
    H = HierarchyStructure(2, 3)
    print(H)
    vecind = [tuple(k) for k in H.vecind]

    i = vecind.index((1, 1))
    assert (1, 1) == tuple(H.vecind[i])
    assert (2, 1) == tuple(H.vecind[H.indab[i, 0]])
    assert (1, 2) == tuple(H.vecind[H.indab[i, 1]])
    assert (0, 1) == tuple(H.vecind[H.indbl[i, 0]])
    assert (1, 0) == tuple(H.vecind[H.indbl[i, 1]])

    i = vecind.index((3, 0))
    assert (3, 0) == tuple(H.vecind[i])
    assert INVALID_INDEX == H.indab[i, 0]
    assert INVALID_INDEX == H.indab[i, 1]
    assert (2, 0) == tuple(H.vecind[H.indbl[i, 0]])
    assert INVALID_INDEX == H.indbl[i, 1]
