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
from pytest import mark
from numpy.testing import assert_array_equal
from lattpy import spatial


@mark.parametrize("r1, r2, result", [
    ([0],       [0],        0.0),
    ([0],       [2],        2.0),
    ([0],       [-2],       2.0),
    ([0, 0],    [-2, 0],    2.0),
    ([0, 0],    [1, 1],     np.sqrt(2)),
    ([0, 0, 0], [1, 1, 1],  np.sqrt(3)),
])
def test_distance(r1, r2, result):
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    assert spatial.distance(r1, r2) == result
    assert spatial.distance(r2, r1) == result


@mark.parametrize("arrays, result", [
    (([1, 3], [2, 4]),           [1, 2, 3, 4]),
    (([1, 4], [2, 5], [3, 6]),   [1, 2, 3, 4, 5, 6]),
    (([[1, 1], [3, 3]], [[2, 2], [4, 4]]), [[1, 1], [2, 2], [3, 3], [4, 4]])
])
def test_interweave(arrays, result):
    assert_array_equal(spatial.interweave(np.array(arrays)), result)


@mark.parametrize("limits, result", [
    (([0, 1], ),       [[0]]),
    (([0, 1], [0, 1]), [[0, 0]]),
    (([0, 2], ),       [[0], [1]]),
    (([0, 2], [0, 1]), [[0, 0], [1, 0]]),
    (([0, 2], [0, 2]), [[0, 0], [0, 1], [1, 0], [1, 1]]),
    (([0, 3], [0, 2]), [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]),
])
def test_vindices(limits, result):
    assert_array_equal(spatial.vindices(limits), result)


@mark.parametrize("stop, result", [
    (1,         [[0]]),
    (3,         [[0], [1], [2]]),
    ((3, 2),    [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
])
def test_vrange_stop(stop, result):
    assert_array_equal(spatial.vrange(stop), result)


@mark.parametrize("start, stop, result", [
    (0, 1,            [[0]]),
    (0, 3,            [[0], [1], [2]]),
    (1, 3,            [[1], [2]]),
    ((0, 0), (3, 2),  [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]),
    ((1, 0), (3, 2),  [[1, 0], [1, 1], [2, 0], [2, 1]])
])
def test_vrange_startstop(start, stop, result):
    assert_array_equal(spatial.vrange(start, stop), result)


@mark.parametrize("vecs, result", [
    ([[1, 0], [0, 1]],      [1., 1.]),
    ([[2, 0], [0, 1]],      [2., 1.]),
    ([[2, 0], [1, 1]],      [2., 1.]),
    ([[2, 0], [-1, 1]],     [3., 1.]),
    ([[2, 0, 0],
      [0, 2, 0],
      [0, 0, 2]],           [2., 2., 2.]),
])
def test_cell_size(vecs, result):
    assert_array_equal(spatial.cell_size(vecs), result)


@mark.parametrize("vecs, result", [
    ([[1, 0], [0, 1]],      1.0),
    ([[2, 0], [0, 1]],      2.0),
    ([[2, 0], [0, 2]],      4.0),
    ([[2, 0, 0],
      [0, 2, 0],
      [0, 0, 2]],           8.0),
])
def test_cell_colume(vecs, result):
    assert spatial.cell_volume(vecs) == result
