# coding: utf-8
"""
Created on 05 Apr 2020
author: Dylan Jones
"""
import numpy as np


class ConfigurationError(Exception):

    def __init__(self, msg='', hint=''):
        if hint:
            msg += f'({hint})'
        super().__init__(msg)


def vrange(axis_ranges):
    """ Return evenly spaced vectors within a given interval.

    Parameters
    ----------
    axis_ranges: array_like
        ranges for each axis.

    Returns
    -------
    vectors: np.ndarray
    """
    axis = np.meshgrid(*axis_ranges)
    grid = np.asarray([np.asarray(a).flatten("F") for a in axis]).T
    n_vecs = list(grid)
    n_vecs.sort(key=lambda x: x[0])
    return n_vecs


def vlinspace(start, stop, n=1000):
    """ Vector linspace

    Parameters
    ----------
    start: array_like
        d-dimensional start-point
    stop: array_like
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


def distance(r1, r2):
    """ Calculates the euclidian distance bewteen two points.

    Parameters
    ----------
    r1: (N) ndarray
        First input point.
    r2: (N) ndarray
        Second input point of matching size.

    Returns
    -------
    distance: float
    """
    return np.sqrt(np.sum((r1 - r2)**2))


def cell_size(vectors):
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


def cell_volume(vectors):
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


def chain(items, cycle=False):
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
