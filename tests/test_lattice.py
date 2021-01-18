# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
import lattpy as lp
from numpy.testing import assert_array_equal, assert_array_almost_equal
from lattpy import Lattice

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


def test_is_reciprocal():
    # Chain
    rvecs = rchain.vectors
    assert chain.is_reciprocal(rvecs)
    assert not chain.is_reciprocal(-1 * rvecs)
    assert not chain.is_reciprocal(+2 * rvecs)
    assert not chain.is_reciprocal(0.5 * rvecs)
    assert not chain.is_reciprocal(0.0 * rvecs)

    # Square
    rvecs = rsquare.vectors
    assert square.is_reciprocal(rvecs)
    assert not square.is_reciprocal(-1 * rvecs)
    assert not square.is_reciprocal(+2 * rvecs)
    assert not square.is_reciprocal(0.5 * rvecs)
    assert not square.is_reciprocal(0.0 * rvecs)

    # Rectangular
    rvecs = rrect.vectors
    assert rect.is_reciprocal(rvecs)
    assert not rect.is_reciprocal(-1 * rvecs)
    assert not rect.is_reciprocal(+2 * rvecs)
    assert not rect.is_reciprocal(0.5 * rvecs)
    assert not rect.is_reciprocal(0.0 * rvecs)

    # Hexagonal
    rvecs = rhexagonal.vectors
    assert hexagonal.is_reciprocal(rvecs)
    assert not hexagonal.is_reciprocal(-1 * rvecs)
    assert not hexagonal.is_reciprocal(+2 * rvecs)
    assert not hexagonal.is_reciprocal(0.5 * rvecs)
    assert not hexagonal.is_reciprocal(0.0 * rvecs)

    # Cubic
    rvecs = rsc.vectors
    assert sc.is_reciprocal(rvecs)
    assert not sc.is_reciprocal(-1 * rvecs)
    assert not sc.is_reciprocal(+2 * rvecs)
    assert not sc.is_reciprocal(0.5 * rvecs)
    assert not sc.is_reciprocal(0.0 * rvecs)

    # Face-centerec-cudic (fcc)
    rvecs = rfcc.vectors
    assert fcc.is_reciprocal(rvecs)
    assert not fcc.is_reciprocal(-1 * rvecs)
    assert not fcc.is_reciprocal(+2 * rvecs)
    assert not fcc.is_reciprocal(0.5 * rvecs)
    assert not fcc.is_reciprocal(0.0 * rvecs)

    # Body-centerec-cudic (bcc)
    rvecs = rbcc.vectors
    assert bcc.is_reciprocal(rvecs)
    assert not bcc.is_reciprocal(-1 * rvecs)
    assert not bcc.is_reciprocal(+2 * rvecs)
    assert not bcc.is_reciprocal(0.5 * rvecs)
    assert not bcc.is_reciprocal(0.0 * rvecs)


def test_reciprocal_vectors():
    # Chain
    expected = rchain.vectors
    actual = chain.reciprocal_vectors()
    assert_array_equal(expected, actual)

    # Square
    expected = rsquare.vectors
    actual = square.reciprocal_vectors()
    assert_array_equal(expected, actual)

    # Rectangular
    expected = rrect.vectors
    actual = rect.reciprocal_vectors()
    assert_array_equal(expected, actual)

    # Hexagonal
    expected = rhexagonal.vectors
    actual = hexagonal.reciprocal_vectors()
    assert_array_almost_equal(expected, actual)

    # Cubic
    expected = rsc.vectors
    actual = sc.reciprocal_vectors()
    assert_array_equal(expected, actual)

    # Face-centerec-cudic (fcc)
    expected = rfcc.vectors
    actual = fcc.reciprocal_vectors()
    assert_array_equal(expected, actual)

    # Body-centerec-cudic (bcc)
    expected = rbcc.vectors
    actual = bcc.reciprocal_vectors()
    assert_array_equal(expected, actual)


def test_reciprocal_vectors_double():
    # Chain
    expected = chain.vectors
    actual = chain.reciprocal_lattice().reciprocal_vectors()
    assert_array_equal(expected, actual)

    # Square
    expected = square.vectors
    actual = square.reciprocal_lattice().reciprocal_vectors()
    assert_array_equal(expected, actual)

    # Rectangular
    expected = rect.vectors
    actual = rect.reciprocal_lattice().reciprocal_vectors()
    assert_array_equal(expected, actual)

    # Hexagonal
    expected = hexagonal.vectors
    actual = hexagonal.reciprocal_lattice().reciprocal_vectors()
    assert_array_equal(expected, actual)

    # Cubic
    expected = sc.vectors
    actual = sc.reciprocal_lattice().reciprocal_vectors()
    assert_array_equal(expected, actual)

    # Face-centerec-cudic (fcc)
    expected = fcc.vectors
    actual = fcc.reciprocal_lattice().reciprocal_vectors()
    assert_array_equal(expected, actual)

    # Body-centerec-cudic (bcc)
    expected = bcc.vectors
    actual = bcc.reciprocal_lattice().reciprocal_vectors()
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


def test_get_position():
    pass


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
