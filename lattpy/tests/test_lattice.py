# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp
from lattpy.utils import SiteOccupiedError, NoAtomsError, NoConnectionsError
from lattpy import Lattice
import lattpy as lp


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
    indices = np.array([[1, 0], [-1, 0]])
    assert_array_equal(latt.get_neighbors(alpha=0, distidx=0), indices)
    # Next nearest neighbors
    indices = np.array([[-2, 0], [2, 0]])
    assert_array_equal(latt.get_neighbors(alpha=0, distidx=1), indices)

    # square
    latt = Lattice.square()
    latt.add_atom()
    latt.add_connections(2)
    # Nearest neighbors
    indices = np.array([[1, 0, 0], [0, -1, 0], [0, 1, 0], [-1, 0, 0]])
    assert_array_equal(latt.get_neighbors(alpha=0, distidx=0), indices)
    # Next nearest neighbors
    indices = np.array([[1, -1, 0], [-1, -1, 0], [-1, 1, 0], [1, 1, 0]])
    assert_array_equal(latt.get_neighbors(alpha=0, distidx=1), indices)


def test_get_neighbor_positions():
    # chain
    latt = Lattice.chain()
    latt.add_atom()
    latt.add_connections(2)
    # Nearest neighbors
    pos = np.array([[1], [-1]])
    assert_array_equal(latt.get_neighbor_positions(alpha=0, distidx=0), pos)
    # Next nearest neighbors
    pos = np.array([[-2], [2]])
    assert_array_equal(latt.get_neighbor_positions(alpha=0, distidx=1), pos)

    # square
    latt = Lattice.square()
    latt.add_atom()
    latt.add_connections(2)
    # Nearest neighbors
    pos = np.array([[1, 0], [0, -1], [0, 1], [-1, 0]])
    assert_array_equal(latt.get_neighbor_positions(alpha=0, distidx=0), pos)
    # Next nearest neighbors
    pos = np.array([[1, -1], [-1, -1], [-1, 1], [1, 1]])
    assert_array_equal(latt.get_neighbor_positions(alpha=0, distidx=1), pos)


def test_get_neighbor_vectors():
    # chain
    latt = Lattice.chain()
    latt.add_atom()
    latt.add_connections(2)
    # Nearest neighbors
    vecs = np.array([[1], [-1]])
    assert_array_equal(latt.get_neighbor_vectors(alpha=0, distidx=0), vecs)
    # Next nearest neighbors
    vecs = np.array([[-2], [2]])
    assert_array_equal(latt.get_neighbor_vectors(alpha=0, distidx=1), vecs)

    # square
    latt = Lattice.square()
    latt.add_atom()
    latt.add_connections(2)
    # Nearest neighbors
    vecs = np.array([[1, 0], [0, -1], [0, 1], [-1, 0]])
    assert_array_equal(latt.get_neighbor_vectors(alpha=0, distidx=0), vecs)
    # Next nearest neighbors
    vecs = np.array([[1, -1], [-1, -1], [-1, 1], [1, 1]])
    assert_array_equal(latt.get_neighbor_vectors(alpha=0, distidx=1), vecs)


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
