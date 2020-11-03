# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from lattpy.data import NeighbourMap


def test_neighbourmap_nonsymmetric():
    neighbours = NeighbourMap(size=3, num_dist=1)
    neighbours.add(0, 1)
    neighbours.add(1, 2)
    neighbours.add_periodic(2, 0)

    assert neighbours.get(0) == {1}
    assert neighbours.get(1) == {2}
    assert neighbours.get(2) == {0}

    assert neighbours[0, 0] == {1}
    assert neighbours[1, 0] == {2}
    assert neighbours[2, 0] == {0}


def test_neighbourmap_symmetric():
    neighbours = NeighbourMap(size=3, num_dist=1)
    neighbours.add(0, 1, symmetric=True)
    neighbours.add(1, 2, symmetric=True)
    neighbours.add_periodic(2, 0, symmetric=True)

    assert neighbours.get(0) == {1, 2}
    assert neighbours.get(1) == {2, 0}
    assert neighbours.get(2) == {1, 0}

    assert neighbours[0, 0] == {1, 2}
    assert neighbours[1, 0] == {2, 0}
    assert neighbours[2, 0] == {1, 0}


def test_neighbourmap_get_periodic():
    neighbours = NeighbourMap(size=3, num_dist=1)
    neighbours.add(0, 1, symmetric=True)
    neighbours.add(1, 2, symmetric=True)
    neighbours.add_periodic(2, 0, symmetric=True)

    assert neighbours.get_periodic(0, distidx=0) == {2}
    assert neighbours.get_periodic(1, distidx=0) == set()
    assert neighbours.get_periodic(2, distidx=0) == {0}


def test_neighbourmap_get_nonperiodic():
    neighbours = NeighbourMap(size=3, num_dist=1)
    neighbours.add(0, 1, symmetric=True)
    neighbours.add(1, 2, symmetric=True)
    neighbours.add_periodic(2, 0, symmetric=True)

    assert neighbours.get_nonperiodic(0, distidx=0) == {1}
    assert neighbours.get_nonperiodic(1, distidx=0) == {0, 2}
    assert neighbours.get_nonperiodic(2, distidx=0) == {1}


def test_neighbourmap_check_symmetry():
    neighbours = NeighbourMap(size=3, num_dist=1)
    neighbours.add(0, 1, symmetric=True)
    neighbours.add(1, 2, symmetric=True)
    neighbours.add_periodic(2, 0, symmetric=True)
    assert neighbours.check_symmetry() is True

    neighbours = NeighbourMap(size=3, num_dist=1)
    neighbours.add(0, 1, symmetric=True)
    neighbours.add(1, 2, symmetric=False)
    neighbours.add_periodic(2, 0, symmetric=True)
    assert neighbours.check_symmetry() is False


def test_neighbourmap_ensure_symmetry():
    neighbours = NeighbourMap(size=3, num_dist=1)
    neighbours.add(0, 1, symmetric=True)
    neighbours.add(1, 2, symmetric=False)
    assert neighbours.check_symmetry() is False
    neighbours.ensure_symmetry()
    assert neighbours.check_symmetry() is True
