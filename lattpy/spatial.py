# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Spatial algorithms and data structures."""

import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, Voronoi
from typing import Iterable, Sequence, Optional, Union
from .utils import ArrayLike, min_dtype, chain
from .plotting import draw_points, draw_vectors, draw_lines, draw_surfaces


__all__ = [
    "distance", "distances", "interweave", "vindices", "vrange", "cell_size",
    "cell_volume", "compute_vectors", "compute_neighbors", "KDTree", "VoronoiTree",
    "WignerSeitzCell", "rx", "ry", "rz", "rotate2d", "rotate3d",
    "build_periodic_translation_vector"
]


def distance(r1: ArrayLike, r2: ArrayLike, decimals: int = None) -> float:
    """ Calculates the euclidian distance bewteen two points.

    Parameters
    ----------
    r1: array_like
        First input point.
    r2: array_like
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


def distances(r1: ArrayLike, r2: ArrayLike, decimals: int = None) -> np.ndarray:
    """ Calculates the euclidian distance between multiple points.

    Parameters
    ----------
    r1: array_like
        First input point.
    r2: array_like
        Second input point of matching size.
    decimals: int, optional
        Optional decimals to round distance to.

    Returns
    -------
    distance: np.ndarray
    """

    r1 = np.atleast_2d(r1)
    r2 = np.atleast_2d(r2)
    dist = np.sqrt(np.sum(np.square(r1 - r2), axis=1))
    if decimals is not None:
        dist = np.round(dist, decimals=decimals)
    return dist


def interweave(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """ Interweaves multiple arrays along the first axis

    Example
    -------
    >>> arr1 = np.array([[1, 1], [3, 3], [5, 5]])
    >>> arr2 = np.array([[2, 2], [4, 4], [6, 6]])
    >>> interweave([arr1, arr2])
    array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])

    Parameters
    ----------
    arrays: (M) Sequence of (N, ...) array_like
        The input arrays to interwave. The shape of all arrays must match.

    Returns
    -------
    interweaved: (M*N, ....) np.ndarray
    """
    shape = list(arrays[0].shape)
    shape[0] = sum(x.shape[0] for x in arrays)
    result = np.empty(shape, dtype=arrays[0].dtype)
    n = len(arrays)
    for i, arr in enumerate(arrays):
        result[i::n] = arr
    return result


def vindices(limits: Iterable[Sequence[int]], sort_axis: Optional[int] = 0,
             dtype: Optional[Union[int, str, np.dtype]] = None) -> np.ndarray:
    """ Return an array representing the indices of a d-dimensional grid.

    Parameters
    ----------
    limits: (D, 2) array_like
        The limits of the indices for each axis.
    sort_axis: int, optional
        Optional axis that is used to sort indices.
    dtype: int or str or np.dtype, optional
        Optional data-type for storing the lattice indices. By default the given limits
        are checked to determine the smallest possible data-type.

    Returns
    -------
    vectors: (N, D) np.ndarray
    """
    if dtype is None:
        dtype = min_dtype(limits, signed=True)
    limits = np.asarray(limits)
    dim = limits.shape[0]

    # Create meshgrid reshape grid to array of indices

    # version 1:
    # axis = np.meshgrid(*(np.arange(*lim, dtype=dtype) for lim in limits))
    # nvecs = np.asarray([np.asarray(a).flatten("F") for a in axis]).T

    # version 2:
    # slices = [slice(lim[0], lim[1], 1) for lim in limits]
    # nvecs = np.mgrid[slices].astype(dtype).reshape(dim, -1).T

    # version 3:
    size = limits[:, 1] - limits[:, 0]
    nvecs = np.indices(size, dtype=dtype).reshape(dim, -1).T + limits[:, 0]

    # Optionally sort indices along given axis
    if sort_axis is not None:
        nvecs = nvecs[np.lexsort(nvecs.T[[sort_axis]])]

    return nvecs


# noinspection PyIncorrectDocstring
def vrange(start=None, *args,
           dtype: Optional[Union[int, str, np.dtype]] = None,
           sort_axis: Optional[int] = 0, **kwargs) -> np.ndarray:
    """Return evenly spaced vectors within a given interval.

    Parameters
    ----------
    start: array_like, optional
        The starting value of the interval. The interval includes this value.
        The default start value is 0.
    stop: array_like
        The end value of the interval.
    step: array_like, optional
        Spacing between values. If `start` and `stop` are sequences and the `step`
        is a scalar the given step size is used for all dimensions of the vectors.
        The default step size is 1.
    sort_axis: int, optional
        Optional axis that is used to sort indices.
    dtype: dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    Returns
    -------
    vectors: (N, D) np.ndarray
    """
    # parse arguments
    if len(args) == 0:
        stop = start
        start = np.zeros_like(stop)
        step = kwargs.get("step", 1.0)
    elif len(args) == 1:
        stop = args[0]
        step = kwargs.get("step", 1.0)
    else:
        stop, step = args

    start = np.atleast_1d(start)
    stop = np.atleast_1d(stop)
    if step is None:
        step = np.ones_like(start)
    elif not hasattr(step, "__len__"):
        step = np.ones_like(start) * step

    # Create grid and reshape to array of vectors
    slices = [slice(i, f, s) for i, f, s in zip(start, stop, step)]
    array = np.mgrid[slices].reshape(len(slices), -1).T
    # Optionally sort array along given axis
    if sort_axis is not None:
        array = array[np.lexsort(array.T[[sort_axis]])]

    return array if dtype is None else array.astype(dtype)


def cell_size(vectors: ArrayLike) -> np.ndarray:
    """ Computes the shape of the box spawned by the given vectors.

    Parameters
    ----------
    vectors: array_like
        The basis vectors defining the cell.

    Returns
    -------
    size: np.ndarray
    """
    max_values = np.max(vectors, axis=0)
    min_values = np.min(vectors, axis=0)
    min_values[min_values > 0] = 0
    return max_values - min_values


def cell_volume(vectors: ArrayLike) -> float:
    r""" Computes the volume of the unit cell defined by the primitive vectors.

    The volume of the unit-cell in two and three dimensions is defined by
    .. math::
        V_{2d} = \abs{a_1 \cross a_2}, \quad V_{3d} = a_1 \cdot \abs{a_2 \cross a_3}

    For higher dimensions the volume is computed using the determinant:
    .. math::
        V_{d} = \sqrt{\det{A A^T}}
    where .math:`A` is the array of vectors.

    Parameters
    ----------
    vectors: array_like
        The basis vectors defining the cell.

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
        v = np.sqrt(np.linalg.det(np.dot(vectors.T, vectors)))
    return abs(v)


def build_periodic_translation_vector(indices, axs):
    limits = np.array([np.min(indices, axis=0), np.max(indices, axis=0)])
    nvec = np.zeros(indices.shape[1] - 1, dtype=np.int64)
    for ax in np.atleast_1d(axs):
        nvec[ax] = np.floor(limits[1][ax]) + 1
    return nvec


def compute_vectors(a: float, b: Optional[float] = None, c: Optional[float] = None,
                    alpha: Optional[float] = None, beta: Optional[float] = None,
                    gamma: Optional[float] = None,
                    decimals: Optional[int] = 0) -> np.ndarray:
    """ Computes lattice vectors by the lengths and angles. """
    if b is None and c is None:
        vectors = [a]
    elif c is None:
        alpha = np.deg2rad(alpha)
        ax = a
        bx = b * np.cos(alpha)
        by = b * np.sin(alpha)
        vectors = np.array([
            [ax, 0],
            [bx, by]
        ])
    else:
        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)
        gamma = np.deg2rad(gamma)
        ax = a
        bx = b * np.cos(gamma)
        by = b * np.sin(gamma)
        cx = c * np.cos(beta)
        cy = (abs(c) * abs(b) * np.cos(alpha) - bx * cx) / by
        cz = np.sqrt(c ** 2 - cx ** 2 - cy ** 2)
        vectors = np.array([
            [ax, 0, 0],
            [bx, by, 0],
            [cx, cy, cz]
        ])
    if decimals:
        vectors = np.round(vectors, decimals=decimals)
    return vectors


# noinspection PyUnresolvedReferences
class KDTree(cKDTree):
    """Simple wrapper of scipy's cKTree with global query settings."""

    def __init__(self, points, k=1, max_dist=np.inf, eps=0., p=2):
        super().__init__(points)
        self.max_dist = max_dist
        self.k = k
        self.p = p
        self.eps = eps

    def query_ball_point(self, x, r):
        return super().query_ball_point(x, r, self.p, self.eps)

    def query_ball_tree(self, other, r):
        return super().query_ball_tree(other, r, self.p, self.eps)

    def query_pairs(self, r):
        return super().query_pairs(r, self.p, self.eps)

    def query(self, x=None, num_jobs=1, decimals=None, include_zero=False,
              compact=True):
        x = self.data if x is None else x
        dists, neighbors = super().query(x, self.k, self.eps, self.p,
                                         self.max_dist, num_jobs)

        # Remove zero-distance neighbors and convert dtype
        if not include_zero and np.all(dists[:, 0] == 0):
            dists = dists[:, 1:]
            neighbors = neighbors[:, 1:]
        neighbors = neighbors.astype(min_dtype(self.n, signed=False))

        # Remove neighbors with distance larger than max_dist
        if self.max_dist < np.inf:
            invalid = dists > self.max_dist
            neighbors[invalid] = self.n
            dists[invalid] = np.inf

        # Remove all invalid columns
        if compact:
            mask = np.any(dists != np.inf, axis=0)
            neighbors = neighbors[:, mask]
            dists = dists[:, mask]

        # Round distances
        if decimals is not None:
            dists = np.round(dists, decimals=decimals)

        return neighbors, dists


def compute_neighbors(positions, k=20, max_dist=np.inf, num_jobs=1, decimals=None,
                      eps=0., include_zero=False, compact=True, x=None):
    # Build tree and query neighbors
    x = positions if x is None else x
    tree = KDTree(positions, k=k, max_dist=max_dist, eps=eps)
    dists, neighbors = tree.query(x, num_jobs, decimals, include_zero, compact)
    return neighbors, dists


class VoronoiTree:

    def __init__(self, points):
        points = np.asarray(points)
        dim = points.shape[1]
        edges = list()
        if dim == 1:
            vertices = points / 2
            idx = np.where((vertices == np.zeros(vertices.shape[1])).all(axis=1))[0]
            vertices = np.delete(vertices, idx)
            vertices = np.atleast_2d(vertices).T
        else:
            vor = Voronoi(points)
            # Save only finite vertices
            vertices = vor.vertices  # noqa
            for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):  # noqa
                simplex = np.asarray(simplex)
                if np.all(simplex >= 0):
                    edges.append(simplex)

        self.dim = dim
        self.points = points
        self.edges = edges
        self.vertices = vertices
        self.tree = cKDTree(points)  # noqa
        self.origin = self.query(np.zeros(dim))

    def query(self, x, k=1, eps=0):
        return self.tree.query(x, k, eps)  # noqa

    def draw(self, ax=None, color="C0", size=3, lw=1, alpha=0.15, point_color="k",
             point_size=3, draw_data=True, points=True, draw=True, fill=True):

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d" if self.dim == 3 else None)

        if draw_data:
            draw_points(ax, self.points, size=point_size, color=point_color)
            if self.dim > 1:
                draw_vectors(ax, self.points, lw=0.5, color=point_color)

        if points:
            draw_points(ax, self.vertices, size=size, color=color)

        if self.dim == 2 and draw:
            segments = np.array([self.vertices[i] for i in self.edges])
            draw_lines(ax, segments, color=color, lw=lw)
        elif self.dim == 3:
            if draw:
                segments = np.array(
                    [self.vertices[np.append(i, i[0])] for i in self.edges]
                )
                draw_lines(ax, segments, color=color, lw=lw)
            if fill:
                surfaces = np.array([self.vertices[i] for i in self.edges])
                draw_surfaces(ax, surfaces, color=color, alpha=alpha)

        if self.dim == 3:
            ax.set_aspect("equal")
        else:
            ax.set_aspect("equal", "box")

        return ax

    def __repr__(self):
        return f"{self.__class__.__name__}(vertices: {len(self.vertices)})"

    def __str__(self):
        return f"vertices:\n{self.vertices}\n" \
               f"egdes:\n{self.edges}"


class WignerSeitzCell(VoronoiTree):

    def __init__(self, points):
        super().__init__(points)
        self._root = self.query(np.zeros(self.dim))[1]

    @property
    def limits(self):
        return np.array([np.min(self.vertices, axis=0),
                         np.max(self.vertices, axis=0)]).T

    @property
    def size(self):
        return self.limits[1] - self.limits[0]

    def check(self, points):
        cells = np.asarray(self.query(points)[1])
        return cells == self._root

    def arange(self, steps, offset=0.):
        limits = self.limits * (1 + offset)
        steps = [steps] * self.dim if not hasattr(steps, "__len__") else steps
        return [np.arange(*lims, step=step) for lims, step in zip(limits, steps)]

    def linspace(self, nums, offset=0.):
        limits = self.limits * (1 + offset)
        nums = [nums] * self.dim if not hasattr(nums, "__len__") else nums
        return [np.linspace(*lims, num=num) for lims, num in zip(limits, nums)]

    def meshgrid(self, nums=None, steps=None, offset=0., check=True):
        if nums is not None:
            grid = np.array(np.meshgrid(*self.linspace(nums, offset)))
        elif steps is not None:
            grid = np.array(np.meshgrid(*self.arange(steps, offset)))
        else:
            raise ValueError("Either the number of points or the step size "
                             "must be specified")

        if check:
            lengths = grid.shape[1:]
            dims = range(len(lengths))
            for item in itertools.product(*[range(n) for n in lengths]):
                point = np.array([grid[d][item] for d in dims])
                if not self.check(point):
                    for d in dims:
                        grid[d][item] = np.nan
        return grid

    def symmetry_points(self):
        origin = np.zeros((1,))
        corners = self.vertices.copy()
        face_centers = None
        if self.dim == 1:
            return origin, corners, None, None
        elif self.dim == 2:
            edge_centers = np.zeros((len(self.edges), 2))
            for i, simplex in enumerate(self.edges):
                p1, p2 = self.vertices[simplex]
                edge_centers[i] = p1 + (p2 - p1) / 2
        elif self.dim == 3:
            edge_centers = list()
            face_centers = list()
            for i, simplex in enumerate(self.edges):
                edges = self.vertices[simplex]
                # compute face centers
                face_centers.append(np.mean(edges, axis=0))
                # compute edge centers
                for p1, p2 in chain(edges, cycle=True):
                    edge_centers.append(p1 + (p2 - p1) / 2)
            edge_centers = np.asarray(edge_centers)
            face_centers = np.asarray(face_centers)
        else:
            raise NotImplementedError()
        return origin, corners, edge_centers, face_centers


def rx(theta: float) -> np.ndarray:
    """X-Rotation matrix."""
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])


def ry(theta: float) -> np.ndarray:
    """Y-Rotation matrix."""
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, +cos]])


def rz(theta: float) -> np.ndarray:
    """Z-Rotation matrix."""
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])


def rot(thetax: float = 0., thetay: float = 0., thetaz: float = 0.) -> np.ndarray:
    """General rotation matrix"""
    r = np.eye(3)
    if thetaz:
        r = np.dot(r, rz(thetaz))
    if thetay:
        r = np.dot(r, ry(thetay))
    if thetax:
        r = np.dot(r, rz(thetax))
    return r


def rotate2d(a, theta, degree=True):
    """Applies the z-rotation matrix to a 2D point"""
    if degree:
        theta = np.deg2rad(theta)
    return np.dot(a, rz(theta)[:2, :2])


def rotate3d(a, thetax=0., thetay=0., thetaz=0., degree=True):
    """Applies the general rotation matrix to a 3D point"""
    if degree:
        thetax = np.deg2rad(thetax)
        thetay = np.deg2rad(thetay)
        thetaz = np.deg2rad(thetaz)
    return np.dot(a, rot(thetax, thetay, thetaz))
