# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp
from lattpy.utils import (
    SiteOccupiedError, NoAtomsError, NoConnectionsError, NotAnalyzedError
)
from lattpy import Lattice, Circle, Atom
import lattpy as lp


atom = Atom()


PI = np.pi
TWOPI = 2 * np.pi

chain = Lattice.chain(a=1.0)
rchain = Lattice(TWOPI)

square = Lattice.square(a=1.0)
rsquare = Lattice(TWOPI * np.eye(2))

rect = Lattice.rectangular(a1=2.0, a2=1.0)
rrect = Lattice(PI * np.array([[1, 0], [0, 2]]))


hexagonal = Lattice.hexagonal(a=1)
rhexagonal = Lattice(np.array([[+2.0943951, +3.62759873],
                               [+2.0943951, -3.62759873]]))
sc = Lattice.sc(a=1.0)
rsc = Lattice(TWOPI * np.eye(3))

fcc = Lattice.fcc(a=1.0)
rfcc = Lattice(TWOPI * np.array([[+1, +1, -1],
                                 [+1, -1, +1],
                                 [-1, +1, +1]]))
bcc = Lattice.bcc(a=1.0)
rbcc = Lattice(TWOPI * np.array([[+1, +1, 0],
                                 [0, -1, +1],
                                 [-1, 0, +1]]))

LATTICES = [chain, square, rect, hexagonal, sc, fcc, bcc]
RLATTICES = [rchain, rsquare, rrect, rhexagonal, rsc, rfcc, rbcc]

STRUCTURES = list()
_latt = lp.simple_chain()
_latt.build(4)
STRUCTURES.append(_latt)
_latt = lp.simple_square()
_latt.build((4, 4))
STRUCTURES.append(_latt)
_latt = lp.simple_cubic()
_latt.build((4, 4, 4))
STRUCTURES.append(_latt)


@st.composite
def structures(draw):
    return draw(st.sampled_from(STRUCTURES))


def assert_elements_equal1d(actual, expected):
    actual = np.unique(actual)
    expected = np.unique(expected)
    assert len(actual) == len(expected)
    return all(np.isin(actual, expected))


def assert_allclose_elements(actual, expected, atol=0., rtol=1e-7):
    assert_allclose(np.sort(actual), np.sort(expected), rtol, atol)


def assert_equal_elements(actual, expected):
    assert_array_equal(np.sort(actual), np.sort(expected))


def test_is_reciprocal():
    for latt, rlatt in zip(LATTICES, RLATTICES):
        rvecs = rlatt.vectors
        assert latt.is_reciprocal(rvecs)
        assert not latt.is_reciprocal(-1 * rvecs)
        assert not latt.is_reciprocal(+2 * rvecs)
        assert not latt.is_reciprocal(0.5 * rvecs)
        assert not latt.is_reciprocal(0.0 * rvecs)


def test_reciprocal_vectors():
    for latt, rlatt in zip(LATTICES, RLATTICES):
        expected = rlatt.vectors
        actual = latt.reciprocal_vectors()
        assert_allclose(expected, actual)


def test_reciprocal_vectors_double():
    for latt in LATTICES:
        expected = latt.vectors
        actual = latt.reciprocal_lattice().reciprocal_vectors()
        assert_array_equal(expected, actual)


def test_translate():
    # Square lattice
    expected = [2.0, 0.0]
    actual = square.translate([2, 0], [0.0, 0.0])
    assert_array_equal(expected, actual)

    expected = [0.0, 2.0]
    actual = square.translate([0, 2], [0.0, 0.0])
    assert_array_equal(expected, actual)

    expected = [1.0, 2.0]
    actual = square.translate([1, 2], [0.0, 0.0])
    assert_array_equal(expected, actual)

    # Rectangular lattice
    expected = [4.0, 0.0]
    actual = rect.translate([2, 0], [0.0, 0.0])
    assert_array_equal(expected, actual)

    expected = [0.0, 2.0]
    actual = rect.translate([0, 2], [0.0, 0.0])
    assert_array_equal(expected, actual)

    expected = [2.0, 2.0]
    actual = rect.translate([1, 2], [0.0, 0.0])
    assert_array_equal(expected, actual)


def test_itranslate():
    # Square lattice
    expected = [2, 0], [0.0, 0.0]
    actual = square.itranslate([2.0, 0.0])
    assert_array_equal(expected, actual)

    expected = [0, 2], [0.0, 0.0]
    actual = square.itranslate([0.0, 2.0])
    assert_array_equal(expected, actual)

    expected = [1, 2], [0.0, 0.0]
    actual = square.itranslate([1.0, 2.0])
    assert_array_equal(expected, actual)

    # Rectangular lattice
    expected = [1, 0], [0.0, 0.0]
    actual = rect.itranslate([2.0, 0.0])
    assert_array_equal(expected, actual)

    expected = [0, 2], [0.0, 0.0]
    actual = rect.itranslate([0.0, 2.0])
    assert_array_equal(expected, actual)

    expected = [1, 1], [0.0, 0.0]
    actual = rect.itranslate([2.0, 1.0])
    assert_array_equal(expected, actual)


def test_brillouin_zone():
    latt = Lattice.square()
    bz = latt.brillouin_zone()

    expected = [[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]]
    assert_array_equal(bz.vertices / np.pi, expected)

    expected = [[0, 1], [0, 2], [1, 3], [2, 3]]
    assert_array_equal(bz.edges, expected)


@given(st.integers(1, 3))
def test_add_atom(dim):
    latt = Lattice(np.eye(dim))

    latt.add_atom()
    assert_array_equal(latt.atom_positions[0], np.zeros(dim))

    with pytest.raises(SiteOccupiedError):
        latt.add_atom()

    with pytest.raises(ValueError):
        latt.add_atom(np.zeros(4))


def test_get_alpha():
    latt = Lattice(np.eye(2))
    at1 = latt.add_atom([0.0, 0.0], atom="A")
    at2 = latt.add_atom([0.5, 0.5], atom="B")

    assert latt.get_alpha("A") == [0]
    assert latt.get_alpha(at1) == 0

    assert latt.get_alpha("B") == [1]
    assert latt.get_alpha(at2) == 1

    latt = Lattice(np.eye(2))
    latt.add_atom([0.0, 0.0], atom="A")
    latt.add_atom([0.5, 0.5], atom="A")
    assert latt.get_alpha("A") == [0, 1]


def test_get_atom():
    latt = Lattice(np.eye(2))
    at1 = latt.add_atom([0.0, 0.0], atom="A")
    at2 = latt.add_atom([0.5, 0.5], atom="B")

    assert latt.get_atom("A") == at1
    assert latt.get_atom(0) == at1

    assert latt.get_atom("B") == at2
    assert latt.get_atom(1) == at2


def test_add_connection():
    latt = Lattice(np.eye(2))
    latt.add_atom([0.0, 0.0], atom="A")
    latt.add_atom([0.5, 0.5], atom="B")

    latt.add_connection("A", "A", 1)
    latt.add_connection("A", "B", 1)
    latt.analyze()

    # Assert neighbor atom index is right
    assert all(latt.get_neighbors(alpha=0, distidx=0)[:, -1] == 1)
    assert all(latt.get_neighbors(alpha=0, distidx=1)[:, -1] == 0)
    assert all(latt.get_neighbors(alpha=1, distidx=0)[:, -1] == 0)


def test_analyze_exceptions():
    latt = Lattice(np.eye(2))
    with pytest.raises(NoAtomsError):
        latt.analyze()

    latt.add_atom()
    with pytest.raises(NoConnectionsError):
        latt.analyze()


@given(hnp.arrays(np.int64, 2, elements=st.integers(0, 10)), st.integers(0, 1))
def test_get_position(nvec, alpha):
    latt = Lattice(np.eye(2))
    latt.add_atom([0.0, 0.0], atom="A")
    latt.add_atom([0.5, 0.5], atom="B")

    pos = latt.translate(nvec, latt.atom_positions[alpha])
    assert_allclose(latt.get_position(nvec, alpha), pos)


@given(hnp.arrays(np.int64, (10, 2), elements=st.integers(0, 10)), st.integers(0, 1))
def test_get_positions(nvecs, alpha):
    latt = Lattice(np.eye(2))
    latt.add_atom([0.0, 0.0], atom="A")
    latt.add_atom([0.5, 0.5], atom="B")

    indices = np.array([[*nvec, alpha] for nvec in nvecs])
    results = latt.get_positions(indices)
    for res, nvec in zip(results, nvecs):
        pos = latt.translate(nvec, latt.atom_positions[alpha])
        assert_allclose(res, pos)


def test_estimate_index():
    # Square lattice
    expected = [2, 0]
    actual = square.estimate_index([2.0, 0.0])
    assert_array_equal(expected, actual)

    expected = [0, 2]
    actual = square.estimate_index([0.0, 2.0])
    assert_array_equal(expected, actual)

    expected = [1, 2]
    actual = square.estimate_index([1.0, 2.0])
    assert_array_equal(expected, actual)

    # Rectangular lattice
    expected = [1, 0]
    actual = rect.estimate_index([2.0, 0.0])
    assert_array_equal(expected, actual)

    expected = [0, 2]
    actual = rect.estimate_index([0.0, 2.0])
    assert_array_equal(expected, actual)

    expected = [1, 1]
    actual = rect.estimate_index([2.0, 1.0])
    assert_array_equal(expected, actual)


def test_get_neighbors():
    # chain
    latt = Lattice.chain()
    latt.add_atom()
    latt.add_connections(2)
    # Nearest neighbors
    expected = np.array([[1, 0], [-1, 0]])
    for idx in latt.get_neighbors(alpha=0, distidx=0):
        assert any((expected == idx).all(axis=1))
    # Next nearest neighbors
    expected = np.array([[-2, 0], [2, 0]])
    for idx in latt.get_neighbors(alpha=0, distidx=1):
        assert any((expected == idx).all(axis=1))

    # square
    latt = Lattice.square()
    latt.add_atom()
    latt.add_connections(2)
    # Nearest neighbors
    expected = np.array([[1, 0, 0], [0, -1, 0], [0, 1, 0], [-1, 0, 0]])
    for idx in latt.get_neighbors(alpha=0, distidx=0):
        assert any((expected == idx).all(axis=1))
    # Next nearest neighbors
    expected = np.array([[1, -1, 0], [-1, -1, 0], [-1, 1, 0], [1, 1, 0]])
    for idx in latt.get_neighbors(alpha=0, distidx=1):
        assert any((expected == idx).all(axis=1))


def test_get_neighbor_positions():
    # chain
    latt = Lattice.chain()
    latt.add_atom()
    latt.add_connections(2)
    # Nearest neighbors
    expected = np.array([[1], [-1]])
    for idx in latt.get_neighbor_positions(alpha=0, distidx=0):
        assert any((expected == idx).all(axis=1))
    # Next nearest neighbors
    expected = np.array([[-2], [2]])
    for idx in latt.get_neighbor_positions(alpha=0, distidx=1):
        assert any((expected == idx).all(axis=1))

    # square
    latt = Lattice.square()
    latt.add_atom()
    latt.add_connections(2)
    # Nearest neighbors
    expected = np.array([[1, 0], [0, -1], [0, 1], [-1, 0]])
    for idx in latt.get_neighbor_positions(alpha=0, distidx=0):
        assert any((expected == idx).all(axis=1))
    # Next nearest neighbors
    expected = np.array([[1, -1], [-1, -1], [-1, 1], [1, 1]])
    for idx in latt.get_neighbor_positions(alpha=0, distidx=1):
        assert any((expected == idx).all(axis=1))


def test_get_neighbor_vectors():
    # chain
    latt = Lattice.chain()
    latt.add_atom()
    latt.add_connections(2)
    # Nearest neighbors
    expected = np.array([[1], [-1]])
    for idx in latt.get_neighbor_vectors(alpha=0, distidx=0):
        assert any((expected == idx).all(axis=1))
    # Next nearest neighbors
    expected = np.array([[-2], [2]])
    for idx in latt.get_neighbor_vectors(alpha=0, distidx=1):
        assert any((expected == idx).all(axis=1))

    # square
    latt = Lattice.square()
    latt.add_atom()
    latt.add_connections(2)
    # Nearest neighbors
    expected = np.array([[1, 0], [0, -1], [0, 1], [-1, 0]])
    for idx in latt.get_neighbor_vectors(alpha=0, distidx=0):
        assert any((expected == idx).all(axis=1))
    # Next nearest neighbors
    expected = np.array([[1, -1], [-1, -1], [-1, 1], [1, 1]])
    for idx in latt.get_neighbor_vectors(alpha=0, distidx=1):
        assert any((expected == idx).all(axis=1))


def test_volume():
    latt = Lattice(np.eye(2))
    latt.add_atom()
    latt.add_connections()
    latt.build((4, 4), primitive=False)
    assert latt.volume() == 25


def test_alpha():
    latt = Lattice(np.eye(2))
    latt.add_atom([0.0, 0.0], "A")
    latt.add_atom([0.5, 0.5], "B")
    latt.add_connection("A", "A", 1)
    latt.add_connection("A", "B", 1)
    latt.analyze()
    latt.build((4, 4), primitive=False)

    assert latt.alpha(0) == 0
    assert latt.alpha(1) == 1
    assert latt.alpha(2) == 0
    assert latt.alpha(3) == 1


def test_atom():
    latt = Lattice(np.eye(2))
    at1 = latt.add_atom([0.0, 0.0], "A")
    at2 = latt.add_atom([0.5, 0.5], "B")
    latt.add_connection("A", "A", 1)
    latt.add_connection("A", "B", 1)
    latt.analyze()
    latt.build((4, 4), primitive=False)

    assert latt.atom(0) == at1
    assert latt.atom(1) == at2
    assert latt.atom(2) == at1
    assert latt.atom(3) == at2


def test_position():
    latt = Lattice(np.eye(2))
    latt.add_atom([0.0, 0.0], "A")
    latt.add_atom([0.5, 0.5], "B")
    latt.add_connection("A", "A", 1)
    latt.add_connection("A", "B", 1)
    latt.analyze()
    latt.build((4, 4), primitive=False)

    assert_array_equal(latt.position(0), [0.0, 0.0])
    assert_array_equal(latt.position(1), [0.5, 0.5])
    assert_array_equal(latt.position(2), [0.0, 1.0])
    assert_array_equal(latt.position(3), [0.5, 1.5])


def test_index_from_position():
    latt = Lattice(np.eye(2))
    latt.add_atom()
    latt.add_connections()
    latt.build((4, 4), primitive=False)
    for i in range(latt.num_sites):
        pos = latt.position(i)
        assert latt.index_from_position(pos) == i


def test_index_from_lattice_index():
    latt = Lattice(np.eye(2))
    latt.add_atom()
    latt.add_connections()
    latt.build((4, 4), primitive=False)
    for i in range(latt.num_sites):
        ind = latt.indices[i]
        assert latt.index_from_lattice_index(ind) == i


def test_get_base_atom_dict():
    latt = Lattice(np.eye(2))
    ata = latt.add_atom([0, 0], atom="A")
    atb = latt.add_atom([0.5, 0], atom="B")
    latt.add_atom([0.5, 0.5], atom="B")
    result = latt.get_base_atom_dict()

    assert len(result) == 2
    assert_array_equal(result[ata], [[0, 0]])
    assert_array_equal(result[atb], [[0.5, 0.], [0.5, 0.5]])


def test_build():
    pass


def test_build_min_neighbors():
    latt = Lattice(np.eye(2))
    latt.add_atom()
    latt.add_connections()
    shape = Circle([0, 0], radius=5)
    latt.build(shape, min_neighbors=2)
    assert_array_equal(latt.data.get_limits(), [[-4., -4.], [4., 4.]])


def test_build_exceptions():
    latt = Lattice(np.eye(2))
    latt.add_atom()

    with pytest.raises(NoConnectionsError):
        latt.build((5, 5))

    latt.add_connections(analyze=False)
    with pytest.raises(NotAnalyzedError):
        latt.build((5, 5))

    latt.analyze()
    with pytest.raises(ValueError):
        latt.build((5, 5, 5))


@given(structures())
def test_nearest_neighbors(latt):
    for i in range(latt.num_sites):
        expected = latt.neighbors(i, distidx=0)
        actual = latt.nearest_neighbors(i)
        assert_equal_elements(actual, expected)

        expected = latt.neighbors(i, distidx=0, unique=True)
        actual = latt.nearest_neighbors(i, unique=True)
        assert_equal_elements(actual, expected)


@given(structures())
def test_iter_neighbors(latt):
    for i in range(latt.num_sites):
        for distidx, actual in latt.iter_neighbors(i):
            expected = latt.neighbors(i, distidx=distidx)
            assert_equal_elements(actual, expected)
        for distidx, actual in latt.iter_neighbors(i, unique=True):
            expected = latt.neighbors(i, distidx=distidx, unique=True)
            assert_equal_elements(actual, expected)


@given(structures())
def test_check_neighbors(latt):
    for i in range(latt.num_sites):
        for distidx, neighbors in latt.iter_neighbors(i):
            # Check neighbors
            for j in neighbors:
                assert latt.check_neighbors(i, j) == distidx
            # Check not neighbors
            for j in np.random.uniform(0, latt.num_sites, size=20):
                if j not in list(neighbors):
                    assert latt.check_neighbors(i, j) is None


def test_compute_connections():
    latt = Lattice(np.eye(2))
    latt.add_atom()
    latt.add_connections()
    latt.build((4, 4), primitive=False)

    latt2 = Lattice(np.eye(2))
    latt2.add_atom()
    latt2.add_connections()
    latt2.build((4, 4), pos=(5, 0), primitive=False)

    pairs, dists = latt.compute_connections(latt2)
    expected = [[20, 0], [21, 1], [22, 2], [23, 3], [24, 4]]
    assert_array_equal(pairs, expected)
    assert np.all(dists) == 1

    latt2.build((4, 4), pos=(5, 1), primitive=False)

    pairs, dists = latt.compute_connections(latt2)
    expected = [[21, 0], [22, 1], [23, 2], [24, 3]]
    assert_array_equal(pairs, expected)
    assert np.all(dists) == 1


def test_periodic_nearest():
    # Lattice chain
    latt = lp.simple_chain()
    latt.build(9)
    latt.set_periodic(0)
    assert 9 in latt.neighbors(0)

    # Square lattice
    latt = lp.simple_square()
    latt.build((4, 4))

    latt.set_periodic(0)
    assert_elements_equal1d(latt.nearest_neighbors(0), [1, 5, 20])
    assert_elements_equal1d(latt.nearest_neighbors(1), [0, 2, 6, 21])
    assert_elements_equal1d(latt.nearest_neighbors(2), [1, 3, 7, 22])
    assert_elements_equal1d(latt.nearest_neighbors(3), [2, 4, 8, 23])
    assert_elements_equal1d(latt.nearest_neighbors(4), [3, 9, 24])
    latt.set_periodic(1)
    assert_elements_equal1d(latt.nearest_neighbors(0), [1, 5, 4])
    assert_elements_equal1d(latt.nearest_neighbors(5), [0, 6, 10, 9])
    assert_elements_equal1d(latt.nearest_neighbors(10), [5, 11, 15, 14])
    assert_elements_equal1d(latt.nearest_neighbors(15), [10, 16, 20, 19])
    assert_elements_equal1d(latt.nearest_neighbors(20), [15, 21, 24])
    # Only check corners for both axis periodic
    latt.set_periodic([0, 1])
    assert_elements_equal1d(latt.nearest_neighbors(0), [1, 5, 20, 24])
    assert_elements_equal1d(latt.nearest_neighbors(4), [3, 9, 20, 24])
    assert_elements_equal1d(latt.nearest_neighbors(20), [0, 15, 21, 24])
    assert_elements_equal1d(latt.nearest_neighbors(24), [4, 19, 20, 23])

    # graphene lattice
    latt = lp.graphene()
    latt.build((5.5, 4.5))

    latt.set_periodic(0)
    assert_elements_equal1d(latt.nearest_neighbors(0), [1, 15])
    assert_elements_equal1d(latt.nearest_neighbors(2), [3, 15, 21])
    assert_elements_equal1d(latt.nearest_neighbors(8), [9, 21, 23])
    latt.set_periodic(1)
    assert_elements_equal1d(latt.nearest_neighbors(1), [0, 4, 16])
    assert_elements_equal1d(latt.nearest_neighbors(6), [5, 7, 17])
    assert_elements_equal1d(latt.nearest_neighbors(7), [6, 14, 22])
    # Only check corners for both axis periodic
    latt.set_periodic([0, 1])
    assert_elements_equal1d(latt.nearest_neighbors(0), [1, 15, 23])
    assert_elements_equal1d(latt.nearest_neighbors(8), [9, 21, 23])


def test_periodic_next_nearest():
    # Lattice chain
    latt = lp.simple_chain(neighbors=2)
    latt.build(9)
    latt.set_periodic(0)
    assert 8 in latt.neighbors(0, distidx=1)

    # Square lattice
    latt = lp.simple_square(neighbors=2)
    latt.build((4, 4))
    latt.set_periodic(0)
    assert_elements_equal1d(latt.neighbors(0, 1), [6, 21])
    assert_elements_equal1d(latt.neighbors(1, 1), [7, 5, 20, 22])
    assert_elements_equal1d(latt.neighbors(2, 1), [8, 6, 21, 23])
    assert_elements_equal1d(latt.neighbors(3, 1), [9, 7, 22, 24])
    assert_elements_equal1d(latt.neighbors(4, 1), [8, 23])
    latt.set_periodic(1)
    assert_elements_equal1d(latt.neighbors(0, 1), [6, 9])
    assert_elements_equal1d(latt.neighbors(5, 1), [1, 4, 11, 14])
    assert_elements_equal1d(latt.neighbors(10, 1), [6, 9, 16, 19])
    assert_elements_equal1d(latt.neighbors(15, 1), [11, 14, 21, 24])
    assert_elements_equal1d(latt.neighbors(20, 1), [16, 19])
    # Only check corners for both axis periodic
    latt.set_periodic([0, 1])
    assert_elements_equal1d(latt.neighbors(0, 1), [6, 9, 21, 24])
    assert_elements_equal1d(latt.neighbors(4, 1), [5, 8, 20, 23])
    assert_elements_equal1d(latt.neighbors(20, 1), [1, 4, 16, 19])
    assert_elements_equal1d(latt.neighbors(23, 1), [0, 3, 15, 18])


def test_remove_periodic():
    latt = lp.simple_chain()
    latt.build(9)
    latt.set_periodic(0)
    latt.set_periodic(None)
    assert 9 not in latt.neighbors(0)


def test_append():
    latt = Lattice(np.eye(2))
    latt.add_atom()
    latt.add_connections()
    latt.build((4, 4), primitive=False)
    latt2 = latt.copy()
    latt.append(latt2)

    assert_elements_equal1d(latt.nearest_neighbors(20), [15, 21, 25])
    assert_elements_equal1d(latt.nearest_neighbors(21), [16, 20, 22, 26])
    assert_elements_equal1d(latt.nearest_neighbors(22), [17, 21, 23, 27])
    assert_elements_equal1d(latt.nearest_neighbors(23), [18, 22, 24, 28])
    assert_elements_equal1d(latt.nearest_neighbors(24), [19, 23, 29])


def test_extend():
    # Only check size, connections are handled by append
    latt = lp.simple_chain()
    latt.build(4, primitive=False)
    latt.extend(2)
    assert latt.num_sites == 8

    latt = lp.simple_square()
    latt.build((4, 4), primitive=False)
    latt.extend(2, ax=0)
    assert_array_equal(latt.shape, (7, 4))

    latt = lp.simple_square()
    latt.build((4, 4), primitive=False)
    latt.extend(2, ax=1)
    assert_array_equal(latt.shape, (4, 7))


def test_repeat():
    # Only check size, connections are handled by append
    latt = lp.simple_chain()
    latt.build(4, primitive=False)
    latt.repeat()
    assert latt.num_sites == 10

    latt = lp.simple_square()
    latt.build((4, 4), primitive=False)
    latt.repeat(1, ax=0)
    assert_array_equal(latt.shape, (9, 4))

    latt = lp.simple_square()
    latt.build((4, 4), primitive=False)
    latt.repeat(1, ax=1)
    assert_array_equal(latt.shape, (4, 9))


def test_to_dict():
    latt = Lattice(np.eye(2))
    latt.add_atom()
    latt.add_connections()
    latt.build((5, 5))
    d = latt.todict()

    expected = [[1., 0.], [0., 1.]]
    assert_array_equal(d["vectors"], expected)

    expected = [[0., 0.]]
    assert_array_equal(d["positions"], expected)

    expected = [[1]]
    assert_array_equal(d["connections"], expected)

    expected = [5., 5.]
    assert_array_equal(d["shape"], expected)


@given(structures())
def test_fromdict(latt):
    d = latt.todict()
    latt2 = Lattice.fromdict(d)

    assert_array_equal(latt.vectors, latt2.vectors)
    assert_array_equal(latt.atom_positions, latt2.atom_positions)
    assert_array_equal(latt._connections, latt2._connections)
    assert_array_equal(latt.num_neighbors, latt2.num_neighbors)


def test_hash():
    latt = Lattice(np.eye(2))
    latt.add_atom(atom=atom)
    latt.add_connections()
    latt.build((5, 5))
    hash1 = latt.__hash__()
    latt.build((6, 5))
    assert latt.__hash__() != hash1


# =========================================================================
# General tests
# =========================================================================

def test_simple_chain():
    latt = lp.simple_chain(a=1, neighbors=1)
    latt.build(3)

    # Check correct building
    assert latt.num_sites == 4

    expected = np.atleast_2d([0.0, 1.0, 2.0, 3.0]).T
    actual = latt.data.positions
    assert_array_equal(expected, actual)

    latt = lp.simple_chain(a=1, neighbors=1)
    latt.build(3)
    # Check neighbors
    expected = [1]
    actual = latt.nearest_neighbors(0)
    assert_array_equal(expected, actual)

    expected = [0, 2]
    actual = latt.nearest_neighbors(1)
    assert_array_equal(expected, sorted(actual))

    # Check periodic boundary conditions
    latt.set_periodic(0)
    expected = [1, 3]
    actual = latt.nearest_neighbors(0)
    assert_array_equal(expected, sorted(actual))

    latt = lp.simple_chain(a=1, neighbors=2)
    latt.build(6)
    expected = [0, 2, 3]
    actual = latt.neighbors(1)
    assert_array_equal(expected, sorted(actual))

    expected = [0, 1, 3, 4]
    actual = latt.neighbors(2)
    assert_array_equal(expected, sorted(actual))


def test_simple_square():
    latt = lp.simple_square(a=1, neighbors=1)
    latt.build((2, 2))

    # Check correct building
    assert latt.num_sites == 9

    # Check nearest neighbors
    expected = [1, 3, 5, 7]
    actual = latt.nearest_neighbors(4)
    assert_array_equal(expected, sorted(actual))

    # Check periodic boundary conditions
    latt.set_periodic(0)
    expected = [1, 3, 6]
    actual = latt.nearest_neighbors(0)
    assert_array_equal(expected, sorted(actual))

    expected = [0, 2, 4, 7]
    actual = latt.nearest_neighbors(1)
    assert_array_equal(expected, sorted(actual))

    latt.set_periodic(1)
    expected = [1, 2, 3]
    actual = latt.nearest_neighbors(0)
    assert_array_equal(expected, sorted(actual))

    # Check next nearest neighbors
    latt = lp.simple_square(a=1, neighbors=2)
    latt.build((2, 2))

    expected = [0, 2, 6, 8]
    actual = latt.neighbors(4, distidx=1)
    assert_array_equal(expected, sorted(actual))


def test_simple_cubic():
    latt = lp.simple_cubic(a=1, neighbors=1)
    latt.build((2, 2, 2))

    # Check correct building
    assert latt.num_sites == 27

    # Check nearest neighbors
    expected = [4, 10, 12, 14, 16, 22]
    actual = latt.nearest_neighbors(13)
    assert_array_equal(expected, sorted(actual))

    expected = [1, 3, 9]
    actual = latt.nearest_neighbors(0)
    assert_array_equal(expected, sorted(actual))

    # Check periodic boundary conditions
    latt.set_periodic(0)
    expected = [1, 3, 9, 18]
    actual = latt.nearest_neighbors(0)
    assert_array_equal(expected, sorted(actual))

    latt.set_periodic(1)
    expected = [1, 3, 6, 9]
    actual = latt.nearest_neighbors(0)
    assert_array_equal(expected, sorted(actual))

    latt.set_periodic(2)
    expected = [1, 2, 3, 9]
    actual = latt.nearest_neighbors(0)
    assert_array_equal(expected, sorted(actual))

    latt.set_periodic([0, 1])
    expected = [1, 3, 6, 9, 18]
    actual = latt.nearest_neighbors(0)
    assert_array_equal(expected, sorted(actual))

    latt.set_periodic([1, 2])
    expected = [1, 2, 3, 6, 9]
    actual = latt.nearest_neighbors(0)
    assert_array_equal(expected, sorted(actual))

    # Check next nearest neighbors
    latt = lp.simple_cubic(a=1, neighbors=2)
    latt.build((2, 2, 2))

    expected = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
    actual = latt.neighbors(13, distidx=1)
    assert_array_equal(expected, sorted(actual))
