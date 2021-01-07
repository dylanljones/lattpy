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
from lattpy import utils


@mark.parametrize("arrays, result", [
    (([1, 3], [2, 4]),           [1, 2, 3, 4]),
    (([1, 4], [2, 5], [3, 6]),   [1, 2, 3, 4, 5, 6]),
    (([[1, 1], [3, 3]], [[2, 2], [4, 4]]), [[1, 1], [2, 2], [3, 3], [4, 4]])
])
def test_interweave(arrays, result):
    assert_array_equal(utils.interweave(np.array(arrays)), result)


@mark.parametrize("limits, result", [
    (([0, 1], ),       [[0]]),
    (([0, 1], [0, 1]), [[0, 0]]),
    (([0, 2], ),       [[0], [1]]),
    (([0, 2], [0, 1]), [[0, 0], [1, 0]]),
    (([0, 2], [0, 2]), [[0, 0], [0, 1], [1, 0], [1, 1]]),
    (([0, 3], [0, 2]), [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]),
])
def test_vindices(limits, result):
    assert_array_equal(utils.vindices(limits), result)


@mark.parametrize("stop, result", [
    (1,         [[0]]),
    (3,         [[0], [1], [2]]),
    ((3, 2),    [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
])
def test_vrange_stop(stop, result):
    assert_array_equal(utils.vrange(stop), result)


@mark.parametrize("start, stop, result", [
    (0, 1,            [[0]]),
    (0, 3,            [[0], [1], [2]]),
    (1, 3,            [[1], [2]]),
    ((0, 0), (3, 2),  [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]),
    ((1, 0), (3, 2),  [[1, 0], [1, 1], [2, 0], [2, 1]])
])
def test_vrange_startstop(start, stop, result):
    assert_array_equal(utils.vrange(start, stop), result)


@mark.parametrize("items, cycle, result", [
    ([0, 1, 2], False, [[0, 1], [1, 2]]),
    ([0, 1, 2],  True, [[0, 1], [1, 2], [2, 0]]),
    (["0", "1", "2"], False, [["0", "1"], ["1", "2"]]),
    (["0", "1", "2"],  True, [["0", "1"], ["1", "2"], ["2", "0"]]),
])
def test_chain_parametrized(items, cycle, result):
    assert_array_equal(utils.chain(items, cycle), result)
