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
from numpy.testing import assert_array_equal, assert_allclose
from lattpy.basis import LatticeBasis

PI = np.pi
TWOPI = 2 * np.pi

chain = LatticeBasis.chain(a=1.0)
rchain = LatticeBasis(TWOPI)

square = LatticeBasis.square(a=1.0)
rsquare = LatticeBasis(TWOPI * np.eye(2))

rect = LatticeBasis.rectangular(a1=2.0, a2=1.0)
rrect = LatticeBasis(PI * np.array([[1, 0], [0, 2]]))


hexagonal = LatticeBasis.hexagonal(a=1)
rhexagonal = LatticeBasis(
    np.array([[+2.0943951, +3.62759873], [+2.0943951, -3.62759873]])
)
sc = LatticeBasis.sc(a=1.0)
rsc = LatticeBasis(TWOPI * np.eye(3))

fcc = LatticeBasis.fcc(a=1.0)
rfcc = LatticeBasis(TWOPI * np.array([[+1, +1, -1], [+1, -1, +1], [-1, +1, +1]]))
bcc = LatticeBasis.bcc(a=1.0)
rbcc = LatticeBasis(TWOPI * np.array([[+1, +1, 0], [0, -1, +1], [-1, 0, +1]]))

LATTICES = [chain, square, rect, hexagonal, sc, fcc, bcc]
RLATTICES = [rchain, rsquare, rrect, rhexagonal, rsc, rfcc, rbcc]


def assert_elements_equal1d(actual, expected):
    actual = np.unique(actual)
    expected = np.unique(expected)
    assert len(actual) == len(expected)
    return all(np.isin(actual, expected))


def assert_allclose_elements(actual, expected, atol=0.0, rtol=1e-7):
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
    latt = LatticeBasis.square()
    bz = latt.brillouin_zone()

    expected = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]
    assert_array_equal(bz.vertices / np.pi, expected)

    expected = [[0, 1], [0, 2], [1, 3], [2, 3]]
    assert_array_equal(bz.edges, expected)
