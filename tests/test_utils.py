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
from lattpy.utils import vrange, vlinspace, chain


def test_vrange():
    ranges = range(1)
    res = vrange(ranges)
    expected = np.array([[0]])
    assert_array_equal(res, expected)

    ranges = range(3)
    res = vrange(ranges)
    expected = np.array([[0, 1, 2]])
    assert_array_equal(res, expected)

    ranges = range(3), range(2)
    res = vrange(ranges)
    expected = np.array([[0, 0], [0, 1],
                         [1, 0], [1, 1],
                         [2, 0], [2, 1]])
    assert_array_equal(res, expected)


def test_vlinspace():
    start, stop = 0, 8
    res = vlinspace(start, stop, 5)
    expected = np.array([[0, 2, 4, 6, 8]]).T
    assert_array_equal(res, expected)

    start, stop = [0, 0], [8, 0]
    res = vlinspace(start, stop, 5)
    expected = np.array([[0, 2, 4, 6, 8], [0, 0, 0, 0, 0]]).T
    assert_array_equal(res, expected)

    start, stop = [0, 0], [0, 8]
    res = vlinspace(start, stop, 5)
    expected = np.array([[0, 0, 0, 0, 0], [0, 2, 4, 6, 8]]).T
    assert_array_equal(res, expected)

    start, stop = [0, 0], [8, 8]
    res = vlinspace(start, stop, 5)
    expected = np.array([[0, 2, 4, 6, 8], [0, 2, 4, 6, 8]]).T
    assert_array_equal(res, expected)


def test_chain():
    items = 0, 1, 2
    res = chain(items, cycle=False)
    expected = [[0, 1], [1, 2]]
    assert_array_equal(res, expected)

    res = chain(items, cycle=True)
    expected = [[0, 1], [1, 2], [2, 0]]
    assert_array_equal(res, expected)


@mark.parametrize("items, cycle, result", [
    ([0, 1, 2], False, [[0, 1], [1, 2]]),
    ([0, 1, 2],  True, [[0, 1], [1, 2], [2, 0]]),
    (["0", "1", "2"], False, [["0", "1"], ["1", "2"]]),
    (["0", "1", "2"],  True, [["0", "1"], ["1", "2"], ["2", "0"]]),
])
def test_chain_parametrized(items, cycle, result):
    assert_array_equal(chain(items, cycle), result)
