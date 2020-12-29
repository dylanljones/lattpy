# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from pytest import mark
from numpy.testing import assert_array_equal
from lattpy.spatial import distance, cell_size, cell_volume


def test_distance():
    r1, r2 = np.array([0]), np.array([0])
    expected = 0.0
    assert distance(r1, r2) == expected
    assert distance(r2, r1) == expected

    r1, r2 = np.array([0]), np.array([2])
    expected = 2.0
    assert distance(r1, r2) == expected
    assert distance(r2, r1) == expected

    r1, r2 = np.array([0, 0]), np.array([2, 0])
    expected = 2.0
    assert distance(r1, r2) == expected
    assert distance(r2, r1) == expected

    r1, r2 = np.array([0, 0]), np.array([-2, 0])
    expected = 2.0
    assert distance(r1, r2) == expected
    assert distance(r2, r1) == expected

    r1, r2 = np.array([0, 0]), np.array([1, 1])
    expected = np.sqrt(2)
    assert distance(r1, r2) == expected
    assert distance(r2, r1) == expected

    r1, r2 = np.array([0, 0, 0]), np.array([1, 1, 1])
    expected = np.sqrt(3)
    assert distance(r1, r2) == expected
    assert distance(r2, r1) == expected


def test_cell_size():
    vecs = np.eye(2).T
    expected = [1, 1]
    res = cell_size(vecs)
    assert_array_equal(res, expected)

    vecs = np.array([[2, 0], [0, 1]])
    expected = [2, 1]
    res = cell_size(vecs)
    assert_array_equal(res, expected)

    vecs = np.array([[2, 0], [1, 1]])
    expected = [2, 1]
    res = cell_size(vecs)
    assert_array_equal(res, expected)

    vecs = np.array([[2, 0], [-1, 1]])
    expected = [3, 1]
    res = cell_size(vecs)
    assert_array_equal(res, expected)


def test_cell_colume():
    # square lattice vectors
    vecs = np.array([[2, 0], [0, 2]])
    res = cell_volume(vecs)
    assert res == 4.0

    # rectangle lattice vectors
    vecs = np.array([[2, 0], [0, 1]])
    res = cell_volume(vecs)
    assert res == 2.0

    # cubic lattice vectors
    vecs = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    res = cell_volume(vecs)
    assert res == 8.0
