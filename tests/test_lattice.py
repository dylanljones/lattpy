# coding: utf-8
"""
Created on 13 May 2020
author: Dylan Jones
"""
import numpy as np
from numpy.testing import assert_array_equal
import lattpy as lp


# =========================================================================
# General tests
# =========================================================================

def test_simple_chain():
    latt = lp.simple_chain(a=1, neighbours=1)
    latt.build(3, inbound=True)

    # Check correct building
    assert latt.n_sites == 4

    expected = np.atleast_2d([0.0, 1.0, 2.0, 3.0]).T
    actual = latt.all_positions()
    assert_array_equal(expected, actual)

    latt = lp.simple_chain(a=1, neighbours=1)
    latt.build(3, inbound=True)
    # Check neighbours
    expected = [1]
    actual = latt.nearest_neighbours(0)
    assert_array_equal(expected, actual)

    expected = [0, 2]
    actual = latt.nearest_neighbours(1)
    assert_array_equal(expected, actual)

    # Check periodic boundary conditions
    latt.set_periodic(0)
    expected = [1, 3]
    actual = latt.nearest_neighbours(0)
    assert_array_equal(expected, actual)

    latt = lp.simple_chain(a=1, neighbours=2)
    latt.build(6, inbound=True)
    expected = [{0, 2}, {3}]
    actual = latt.data.neighbours[1]
    assert_array_equal(expected, actual)

    expected = [{1, 3}, {0, 4}]
    actual = latt.data.neighbours[2]
    assert_array_equal(expected, actual)


def test_simple_square():
    latt = lp.simple_square(a=1, neighbours=1)
    latt.build((2, 2), inbound=True)

    # Check correct building
    assert latt.n_sites == 9

    # Check nearest neighbours
    expected = [1, 3, 5, 7]
    actual = latt.nearest_neighbours(4)
    assert_array_equal(expected, actual)

    # Check periodic boundary conditions
    latt.set_periodic(0)
    expected = [1, 3, 6]
    actual = latt.nearest_neighbours(0)
    assert_array_equal(expected, actual)

    expected = [0, 2, 4, 7]
    actual = latt.nearest_neighbours(1)
    assert_array_equal(expected, actual)

    latt.set_periodic(1)
    expected = [1, 3, 2]
    actual = latt.nearest_neighbours(0)
    assert_array_equal(expected, actual)

    # Check next nearest neighbours
    latt = lp.simple_square(a=1, neighbours=2)
    latt.build((2, 2), inbound=True)

    expected = [0, 8, 2, 6]
    actual = latt.neighbours(4, distidx=1)
    assert_array_equal(expected, actual)


def test_simple_cubic():
    latt = lp.simple_cubic(a=1, neighbours=1)
    latt.build((2, 2, 2), inbound=True)

    # Check correct building
    assert latt.n_sites == 27

    # Check nearest neighbours
    expected = [4, 10, 12, 14, 16, 22]
    actual = latt.nearest_neighbours(13)
    assert_array_equal(expected, actual)

    expected = [1, 3, 9]
    actual = latt.nearest_neighbours(0)
    assert_array_equal(expected, actual)

    # Check periodic boundary conditions
    latt.set_periodic(0)
    expected = [1, 3, 9, 18]
    actual = latt.nearest_neighbours(0)
    assert_array_equal(expected, actual)

    latt.set_periodic(1)
    expected = [1, 3, 9, 2]
    actual = latt.nearest_neighbours(0)
    assert_array_equal(expected, actual)

    latt.set_periodic(2)
    expected = [1, 3, 9, 6]
    actual = latt.nearest_neighbours(0)
    assert_array_equal(expected, actual)

    latt.set_periodic([0, 1])
    expected = [1, 3, 9, 18, 2]
    actual = latt.nearest_neighbours(0)
    assert_array_equal(expected, actual)

    # Check next nearest neighbours
    latt = lp.simple_cubic(a=1, neighbours=2)
    latt.build((2, 2, 2), inbound=True)

    expected = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
    actual = latt.neighbours(13, distidx=1)
    assert_array_equal(expected, actual)
