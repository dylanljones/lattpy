# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import math
import numpy as np
from typing import Iterable, List, Sequence, Optional, Union


def vrange(axis_ranges: Iterable) -> List:
    """ Return evenly spaced vectors within a given interval.

    Parameters
    ----------
    axis_ranges: array_like
        ranges for each axis.

    Returns
    -------
    vectors: list
    """
    axis = np.meshgrid(*axis_ranges)
    grid = np.asarray([np.asarray(a).flatten("F") for a in axis]).T
    n_vecs = list(grid)
    n_vecs.sort(key=lambda x: x[0])
    return n_vecs


def vlinspace(start: Union[float, Sequence[float]],
              stop: Union[float, Sequence[float]],
              n: Optional[int] = 1000) -> np.ndarray:
    """ Vector linspace

    Parameters
    ----------
    start: array_like or float
        d-dimensional start-point
    stop: array_like or float
        d-dimensional stop-point
    n: int, optional
        number of points, default=1000

    Returns
    -------
    vectors: np.ndarray
    """
    start = np.atleast_1d(start)
    stop = np.atleast_1d(stop)
    if not hasattr(start, '__len__') and not hasattr(stop, '__len__'):
        return np.linspace(start, stop, n)
    axes = [np.linspace(start[i], stop[i], n) for i in range(len(start))]
    return np.asarray(axes).T


def distance(r1: np.ndarray, r2: np.ndarray, decimals: Optional[int] = None) -> float:
    """ Calculates the euclidian distance bewteen two points.

    Parameters
    ----------
    r1: (N) ndarray
        First input point.
    r2: (N) ndarray
        Second input point of matching size.
    decimals: int, optional
        Optional decimals to round distance to.

    Returns
    -------
    distance: float
    """
    dist = math.sqrt(np.sum(np.square(r1 - r2)))
    if decimals is not None:
        dist = round(dist, decimals)
    return dist


def cell_size(vectors: np.ndarray) -> np.ndarray:
    """ Computes the shape of the box spawned by the given vectors.

    Parameters
    ----------
    vectors: (N, N) array_like

    Returns
    -------
    size: np.ndarray
    """
    max_values = np.max(vectors, axis=0)
    min_values = np.min(vectors, axis=0)
    min_values[min_values > 0] = 0
    return max_values - min_values


def cell_volume(vectors: np.ndarray) -> float:
    r""" Computes the volume of the unit cell defined by the primitive vectors.

    The volume of the unit-cell in two and three dimensions is defined by
    .. math::
        V_{2d} = \abs{a_1 \cross a_2}, \quad V_{3d} = a_1 \cdot \abs{a_2 \cross a_3}

    Returns
    -------
    vol: float
    """
    dim = len(vectors)
    if dim == 1:
        v = float(vectors)
    elif dim == 2:
        v = np.cross(vectors[0], vectors[1])
    elif dim == 3:
        cross = np.cross(vectors[1], vectors[2])
        v = np.dot(vectors[0], cross)
    else:
        raise ValueError('Only 1, 2 or 3D cells supported!')
    return abs(v)


def chain(items: Sequence, cycle: bool = False) -> List:
    """ Create chain between items

    Parameters
    ----------
    items: array_like
        items to join to chain
    cycle: bool, optional
        cycle to the start of the chain if True, default: False

    Returns
    -------
    chain: list
        chain of items

    Example
    -------
    >>> print(chain(["x", "y", "z"]))
    [['x', 'y'], ['y', 'z']]

    >>> print(chain(["x", "y", "z"], True))
    [['x', 'y'], ['y', 'z'], ['z', 'x']]
    """
    result = list()
    for i in range(len(items)-1):
        result.append([items[i], items[i+1]])
    if cycle:
        result.append([items[-1], items[0]])
    return result


class VectorBasis:

    def __init__(self, vectors: Union[int, float, Sequence[Sequence[float]]]):
        # Transpose vectors so they are a column of the basis matrix
        vectors = np.atleast_2d(vectors).T

        self.dim: int = len(vectors)
        self._vectors: np.ndarray = vectors
        self._vectors_inv: np.ndarray = np.linalg.inv(self._vectors)
        self.cell_size: np.ndarray = cell_size(vectors)
        self.cell_volume: float = cell_volume(vectors)

    @property
    def vectors(self) -> np.ndarray:
        """ (N, N) np.ndarray: Array with basis vectors as rows"""
        return self._vectors.T

    @property
    def vectors3d(self) -> np.ndarray:
        """ (3, 3) np.ndarray: Basis vectors expanded to three dimensions """
        vectors = np.eye(3)
        vectors[:self.dim, :self.dim] = self._vectors
        return vectors.T

    def transform(self, world_coords) -> np.ndarray:
        """ Transform the world-coordinates (x, y, ...) into the basis coordinates (n, m, ...)

        Parameters
        ----------
        world_coords: (N) array_like

        Returns
        -------
        basis_coords: (N) np.ndarray
        """
        return self._vectors_inv @ np.asarray(world_coords)

    def itransform(self, basis_coords: Sequence) -> np.ndarray:
        """ Transform the basis-coordinates (n, m, ...) into the world coordinates (x, y, ...)

        Parameters
        ----------
        basis_coords: (N) array_like

        Returns
        -------
        world_coords: (N) np.ndarray
        """
        return self._vectors @ np.asarray(basis_coords)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dim}D)"

    def __str__(self) -> str:
        sep = "  "
        lines = [self.__repr__()]
        for i in range(self.dim):
            parts = list()
            for j in range(self.dim):
                parts.append(f"[{self.vectors[i, j]:.1f}]")
            lines.append(sep.join(parts))
        return "\n".join(lines)
