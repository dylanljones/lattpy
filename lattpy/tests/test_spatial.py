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
from lattpy import spatial, simple_chain, simple_square, simple_cubic


finite_floats = st.floats(allow_nan=False, allow_infinity=False)


@given(hnp.arrays(np.float64, 10, elements=finite_floats),
       hnp.arrays(np.float64, 10, elements=finite_floats))
def test_distance(a, b):
    expected = math.sqrt(np.sum(np.square(a - b)))

    res = spatial.distance(a, b)
    assert res == expected

    res = spatial.distance(a, b, decimals=3)
    assert res == round(expected, 3)


@given(hnp.arrays(np.float64, (5, 10), elements=finite_floats),
       hnp.arrays(np.float64, (5, 10), elements=finite_floats))
def test_distances(a, b):
    results = spatial.distances(a, b)
    for i in range(len(results)):
        expected = math.sqrt(np.sum(np.square(a[i] - b[i])))
        assert results[i] == expected

    results = spatial.distances(a, b, 3)
    for i in range(len(results)):
        expected = math.sqrt(np.sum(np.square(a[i] - b[i])))
        assert results[i] == np.round(expected, 3)


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


def test_wignerseitz_symmetry_points():
    # Test 1D
    latt = simple_chain()
    ws = latt.wigner_seitz_cell()
    origin, corners, edge_centers, face_centers = ws.symmetry_points()
    assert_array_equal(origin, [0.0])
    assert_array_equal(corners, [[-0.5], [0.5]])
    assert edge_centers is None
    assert face_centers is None

    # Test 2D
    latt = simple_square()
    ws = latt.wigner_seitz_cell()
    origin, corners, edge_centers, face_centers = ws.symmetry_points()

    assert_array_equal(origin, [0.0, 0.0])
    assert_array_equal(2 * corners, [[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])
    assert_array_equal(2 * edge_centers, [[0., -1.], [-1., 0.], [1., 0.], [0., 1.]])
    assert face_centers is None

    # Test 3D
    latt = simple_cubic()
    ws = latt.wigner_seitz_cell()
    origin, corners, edge_centers, face_centers = ws.symmetry_points()

    c = [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ]
    e = [
        [-0.5, -0.5, 0.0],
        [0.0, -0.5, -0.5],
        [0.5, -0.5, 0.0],
        [0.0, -0.5, 0.5],
        [-0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, -0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [-0.5, 0.0, 0.5],
        [0.0, -0.5, 0.5],
        [-0.5, -0.5, 0.0],
        [-0.5, 0.0, -0.5],
        [-0.5, 0.5, 0.0],
        [-0.5, 0.0, 0.5],
        [0.5, 0.0, -0.5],
        [0.0, 0.5, -0.5],
        [-0.5, 0.0, -0.5],
        [0.0, -0.5, -0.5],
        [0.5, -0.5, 0.0],
        [0.5, 0.0, -0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
    ]

    f = [
        [0.0, -0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [-0.5, 0.0, 0.0],
        [0.0, 0.0, -0.5],
        [0.5, 0.0, 0.0],
    ]

    assert_array_equal(origin, [0., 0., 0.])
    assert_array_equal(corners, c)
    assert_array_equal(edge_centers, e)
    assert_array_equal(face_centers, f)


def test_compute_vectors():
    # Test square vectors
    vecs = spatial.compute_vectors(1.0, 1.0, alpha=90)
    assert_allclose(vecs, np.eye(2), atol=1e-16)

    # Text hexagonal vectors
    vecs = spatial.compute_vectors(1.0, 1.0, alpha=60)
    expected = np.array([[1, 0], [0.5, math.sqrt(3)/2]])
    assert_allclose(vecs, expected, atol=1e-16)

    # Test cubic vectors
    vecs = spatial.compute_vectors(1.0, 1.0, 1.0, alpha=90, beta=90, gamma=90)
    assert_allclose(vecs, np.eye(3), atol=1e-16)


def test_rx():
    expected = np.eye(3)
    assert_allclose(spatial.rx(0), expected)

    expected = [[1., 0., 0.],
                [0., 0.70710678, -0.70710678],
                [0., 0.70710678, 0.70710678]]
    assert_allclose(spatial.rx(np.pi/4), expected)

    expected = [[1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]]
    assert_allclose(spatial.rx(np.pi/2), expected, atol=1e-10)

    expected = [[1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]]
    assert_allclose(spatial.rx(np.pi), expected, atol=1e-10)


def test_ry():
    expected = np.eye(3)
    assert_allclose(spatial.ry(0), expected)

    expected = [[0.70710678, 0., 0.70710678],
                [0.,  1.,  0.],
                [-0.70710678, 0., 0.70710678]]
    assert_allclose(spatial.ry(np.pi/4), expected)

    expected = np.array([[0, 0, 1],
                         [0, 1, 0],
                         [-1, 0, 0]])
    assert_allclose(spatial.ry(np.pi/2), expected, atol=1e-10)

    expected = [[-1, 0, 0],
                [0, 1, 0],
                [0, 0, -1]]
    assert_allclose(spatial.ry(np.pi), expected, atol=1e-10)


def test_rz():
    expected = np.eye(3)
    assert_allclose(spatial.rz(0), expected)

    expected = [[0.70710678, -0.70710678, 0.],
                [0.70710678, 0.70710678, 0.],
                [0., 0.,  1.]]
    assert_allclose(spatial.rz(np.pi/4), expected)

    expected = np.array([[0, -1, 0],
                         [1, 0, 0],
                         [0, 0, 1]])
    assert_allclose(spatial.rz(np.pi/2), expected, atol=1e-10)

    expected = [[-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]]
    assert_allclose(spatial.rz(np.pi), expected, atol=1e-10)
