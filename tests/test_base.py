# coding: utf-8
"""
Created on 22 Apr 2020
author: Dylan Jones
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from lattpy.bravais import BravaisLattice

PI = np.pi
TWOPI = 2 * np.pi

chain = BravaisLattice.chain(a=1.0)
rchain = BravaisLattice(TWOPI)

square = BravaisLattice.square(a=1.0)
rsquare = BravaisLattice(TWOPI * np.eye(2))

rect = BravaisLattice.rectangular(a1=2.0, a2=1.0)
rrect = BravaisLattice(PI * np.array([[1, 0], [0, 2]]))


hexagonal = BravaisLattice.hexagonal(a=1)
rhexagonal = BravaisLattice(np.array([[+2.0943951, +3.62759873],
                                      [+2.0943951, -3.62759873]]))
sc = BravaisLattice.sc(a=1.0)
rsc = BravaisLattice(TWOPI * np.eye(3))

fcc = BravaisLattice.fcc(a=1.0)
rfcc = BravaisLattice(TWOPI * np.array([[+1, +1, -1],
                                        [+1, -1, +1],
                                        [-1, +1, +1]]))
bcc = BravaisLattice.bcc(a=1.0)
rbcc = BravaisLattice(TWOPI * np.array([[+1, +1, 0],
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
