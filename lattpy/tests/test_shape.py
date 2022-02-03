# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import math
import numpy as np
from pytest import mark
from numpy.testing import assert_array_equal, assert_allclose
from hypothesis import given, settings, assume, strategies as st
import hypothesis.extra.numpy as hnp
from lattpy import shape


dim = st.shared(st.integers(1, 3), key="d")


@given(
    hnp.arrays(np.float64, dim, elements=st.floats(0.1, 100)),
    hnp.arrays(np.float64, dim, elements=st.floats(-100, 100)))
def test_shape(size, pos):
    s = shape.Shape(size, pos)

    d = len(pos)
    limits = (pos + np.array([np.zeros(d), size])).T
    assert_array_equal(s.limits(), limits)

    pts = np.random.uniform(-np.min(limits) - 10, +np.max(limits) + 10, size=(100, d))
    mask = s.contains(pts, tol=0)
    for point, res in zip(pts, mask):
        expected = True
        for i in range(d):
            expected = expected and (limits[i, 0] <= point[i] <= limits[i, 1])
        assert res == expected


@given(hnp.arrays(np.float64, dim, elements=st.floats(0.1, 100)))
def test_shape_basis(size):
    basis = np.eye(len(size))
    basis[0, 0] = 2
    s = shape.Shape(size, basis=basis)

    d = len(size)
    size[0] = size[0] * 2
    limits = (np.array([np.zeros(d), size])).T
    assert_allclose(s.limits(), limits, atol=1e-10)

    pts = np.random.uniform(-np.min(limits) - 10, +np.max(limits) + 10, size=(100, d))
    mask = s.contains(pts)
    for point, res in zip(pts, mask):
        expected = True
        for i in range(d):
            expected = expected and (limits[i, 0] <= point[i] <= limits[i, 1])
        assert res == expected


@given(hnp.arrays(np.float64, 2, elements=st.floats(-100, 100)), st.floats(1, 100))
def test_circle(pos, radius):
    s = shape.Circle(pos, radius)

    limits = np.array([pos - radius, pos + radius]).T
    assert_array_equal(s.limits(), limits)

    pts = np.random.uniform(-np.min(limits) - 10, +np.max(limits) + 10, size=(100, 2))
    mask = s.contains(pts)
    for point, res in zip(pts, mask):
        diff = point - pos
        dist = math.sqrt(diff[0]*diff[0] + diff[1]*diff[1])
        assert res == (dist <= radius)


@given(
    hnp.arrays(np.float64, 2, elements=st.floats(-100, 100)),
    st.floats(50, 100),
    st.floats(0, 49)
)
def test_donut(pos, outer, inner):
    s = shape.Donut(pos, outer, inner)

    limits = np.array([pos - outer, pos + outer]).T
    assert_array_equal(s.limits(), limits)

    pts = np.random.uniform(-np.min(limits) - 10, +np.max(limits) + 10, size=(100, 2))
    mask = s.contains(pts)
    for point, res in zip(pts, mask):
        diff = point - pos
        dist = math.sqrt(diff[0]*diff[0] + diff[1]*diff[1])
        assert res == (inner <= dist <= outer)


def test_convex_hull_2d():
    points = np.array([[0, 0], [2, 0], [2, 1], [1, 1]])
    s = shape.ConvexHull(points)

    limits = np.array([[0, 2], [0, 1]])
    assert_array_equal(s.limits(), limits)

    pts = np.random.uniform(-np.min(limits) - 1, +np.max(limits) + 1, size=(100, 2))
    mask = s.contains(pts)
    for point, res in zip(pts, mask):
        x, y = point
        if x <= 1:
            # Left half: triangle
            expected = (0 <= x <= 1) and (0 <= y <= x)
        else:
            # Right: square
            expected = (1 <= x <= 2) and (0 <= y <= 1)
        assert res == expected
