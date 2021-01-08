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
from lattpy import utils


@mark.parametrize("a, signed, result", [
    (+127,          False,     np.uint8),
    (-127,          False,     np.int8),
    (+127,          True,      np.int8),
    (+128,          False,     np.uint8),
    (-128,          False,     np.int8),
    (+128,          True,      np.int16),
    ([-128, 127],   False,     np.int8),
    ([-128, 128],   False,     np.int16),
    ([-129, 127],   False,     np.int16),
    ([+30,  127],   False,     np.uint8),
])
def test_min_dtype(a, signed, result):
    assert utils.min_dtype(a, signed) == result


@mark.parametrize("items, cycle, result", [
    ([0, 1, 2], False, [[0, 1], [1, 2]]),
    ([0, 1, 2],  True, [[0, 1], [1, 2], [2, 0]]),
    (["0", "1", "2"], False, [["0", "1"], ["1", "2"]]),
    (["0", "1", "2"],  True, [["0", "1"], ["1", "2"], ["2", "0"]]),
])
def test_chain(items, cycle, result):
    assert_array_equal(utils.chain(items, cycle), result)
