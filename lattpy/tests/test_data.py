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
import hypothesis.extra.numpy as hnp
from lattpy.utils import SiteOccupiedError, NoAtomsError, NoConnectionsError
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


