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
from lattpy.utils import SiteOccupiedError, NoAtomsError, NoConnectionsError
from lattpy import Atom
from lattpy.structure import LatticeStructure

atom = Atom()


@given(st.integers(1, 3))
def test_add_atom(dim):
    latt = LatticeStructure(np.eye(dim))

    latt.add_atom()
    assert_array_equal(latt.atom_positions[0], np.zeros(dim))

    with pytest.raises(SiteOccupiedError):
        latt.add_atom()

    with pytest.raises(ValueError):
        latt.add_atom(np.zeros(4))


def test_get_alpha():
    latt = LatticeStructure(np.eye(2))
    at1 = latt.add_atom([0.0, 0.0], atom="A")
    at2 = latt.add_atom([0.5, 0.5], atom="B")

    assert latt.get_alpha("A") == [0]
    assert latt.get_alpha(at1) == 0

    assert latt.get_alpha("B") == [1]
    assert latt.get_alpha(at2) == 1

    latt = LatticeStructure(np.eye(2))
    latt.add_atom([0.0, 0.0], atom="A")
    latt.add_atom([0.5, 0.5], atom="A")
    assert latt.get_alpha("A") == [0, 1]


def test_get_atom():
    latt = LatticeStructure(np.eye(2))
    at1 = latt.add_atom([0.0, 0.0], atom="A")
    at2 = latt.add_atom([0.5, 0.5], atom="B")

    assert latt.get_atom("A") == at1
    assert latt.get_atom(0) == at1

    assert latt.get_atom("B") == at2
    assert latt.get_atom(1) == at2


def test_add_connection():
    latt = LatticeStructure(np.eye(2))
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
    latt = LatticeStructure(np.eye(2))
    with pytest.raises(NoAtomsError):
        latt.analyze()

    latt.add_atom()
    with pytest.raises(NoConnectionsError):
        latt.analyze()


@given(hnp.arrays(np.int64, 2, elements=st.integers(0, 10)), st.integers(0, 1))
def test_get_position(nvec, alpha):
    latt = LatticeStructure(np.eye(2))
    latt.add_atom([0.0, 0.0], atom="A")
    latt.add_atom([0.5, 0.5], atom="B")

    pos = latt.translate(nvec, latt.atom_positions[alpha])
    assert_allclose(latt.get_position(nvec, alpha), pos)


@given(hnp.arrays(np.int64, (10, 2), elements=st.integers(0, 10)), st.integers(0, 1))
def test_get_positions(nvecs, alpha):
    latt = LatticeStructure(np.eye(2))
    latt.add_atom([0.0, 0.0], atom="A")
    latt.add_atom([0.5, 0.5], atom="B")

    indices = np.array([[*nvec, alpha] for nvec in nvecs])
    results = latt.get_positions(indices)
    for res, nvec in zip(results, nvecs):
        pos = latt.translate(nvec, latt.atom_positions[alpha])
        assert_allclose(res, pos)


def test_estimate_index():
    # Square lattice
    square = LatticeStructure.square()

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
    rect = LatticeStructure.rectangular(a1=2.0, a2=1.0)

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
    latt = LatticeStructure.chain()
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
    latt = LatticeStructure.square()
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
    latt = LatticeStructure.chain()
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
    latt = LatticeStructure.square()
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
    latt = LatticeStructure.chain()
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
    latt = LatticeStructure.square()
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


def test_get_base_atom_dict():
    latt = LatticeStructure(np.eye(2))
    ata = latt.add_atom([0, 0], atom="A")
    atb = latt.add_atom([0.5, 0], atom="B")
    latt.add_atom([0.5, 0.5], atom="B")
    result = latt.get_base_atom_dict()

    assert len(result) == 2
    assert_array_equal(result[ata], [[0, 0]])
    assert_array_equal(result[atb], [[0.5, 0.0], [0.5, 0.5]])
