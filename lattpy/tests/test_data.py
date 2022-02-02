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
from hypothesis import given, assume, strategies as st
from lattpy import simple_square, simple_chain


def _construct_lattices():
    latts = list()

    latt = simple_chain()
    latt.build(10)
    latts.append(latt)

    latt = simple_square()
    latt.build((5, 5))
    latts.append(latt)

    return latts


LATTICES = _construct_lattices()


@st.composite
def lattices(draw):
    return draw(st.sampled_from(LATTICES))


@given(lattices())
def test_dim(latt):
    assert latt.dim == latt.data.dim


@given(lattices())
def test_dim(latt):
    assert len(latt.data.positions) == latt.data.num_sites


def test_num_distances():
    chain, square = LATTICES

    assert chain.data.num_distances == 2
    assert square.data.num_distances == 2


@given(lattices())
def test_copy(latt):
    copy = latt.copy()
    copy.data.remove(0)
    assert copy.data.num_sites == latt.data.num_sites - 1


@given(st.integers(0, 100))
def test_remove(site):
    latt = simple_square()
    latt.build((10, 10))
    assume(1 < site < latt.num_sites - 1)

    # find neighbor site lower than the one to remove:
    i = 0
    for i in latt.nearest_neighbors(site):
        if i < site:
            break
    assume(i > 0)
    assume(site in list(latt.nearest_neighbors(i)))

    # Get the number of neighbors of the neighbor site of `site`.
    num_neighbors = len(latt.nearest_neighbors(i))
    # Delete the site
    latt.data.remove(site)

    # check that the number of neighbors is one less than before
    assert len(latt.nearest_neighbors(i)) == (num_neighbors - 1)


def test_sort():
    latt = simple_square()
    latt.build((5, 5))

    assert np.max(np.diff(latt.data.indices, axis=0)[:, 0]) == 1

    latt.data.sort(ax=1)
    assert np.max(np.diff(latt.data.indices, axis=0)[:, 1]) == 1


def test_append():
    latt = simple_square()
    latt.build((4, 4))
    num_sites_original = latt.num_sites
    latt2 = simple_square()
    latt2.build((5, 4), pos=(5, 0))

    latt.data.append(latt2.data)

    original_pos = latt2.positions.copy()
    pos = latt.positions[num_sites_original:]
    assert_array_equal(pos, original_pos)


def test_datamap():
    # Chain
    latt = simple_chain()
    latt.build(5)

    dmap = latt.data.map()
    res = [True, True, True, True, True, True, False, False, False,
           False, False, False, False, False, False, False]
    assert dmap.size == len(res)
    assert_array_equal(dmap.onsite(0), res)

    res = [False, False, False, False, False, False, True, True, True,
           True, True, True, True, True, True, True]
    assert_array_equal(dmap.hopping(0), res)

    data = np.zeros(dmap.size)
    dmap.fill(data, hop=1, eps=2)

    res = np.zeros(dmap.size)
    res[:latt.num_sites] = 2.0
    res[latt.num_sites:] = 1.0
    assert_array_equal(data, res)


def test_site_mask():
    latt = simple_square()
    latt.build((4, 4))
    data = latt.data

    expected = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert_array_equal(data.site_mask([1, 0]), expected)

    expected = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
                1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
    assert_array_equal(data.site_mask([0, 1]), expected)


def test_find_sites():
    latt = simple_square()
    latt.build((4, 4))
    data = latt.data
    mins, maxs = [1, 1], [3, 3]

    expected = [6, 7, 8, 11, 12, 13, 16, 17, 18]
    assert_array_equal(data.find_sites(mins, maxs), expected)

    expected = [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]
    assert_array_equal(data.find_sites(mins, maxs, invert=True), expected)


def test_find_outer_sites():
    latt = simple_square()
    latt.build((4, 4))
    data = latt.data
    offset = 1

    expected = [0, 1, 2, 3, 4, 20, 21, 22, 23, 24]
    assert_array_equal(data.find_outer_sites(0, offset), expected)

    expected = [0, 4, 5, 9, 10, 14, 15, 19, 20, 24]
    assert_array_equal(data.find_outer_sites(1, offset), expected)
