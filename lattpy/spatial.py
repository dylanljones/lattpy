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
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, Voronoi
from typing import Optional
from .utils import chain
from .plotting import draw_points, draw_vectors, draw_lines, draw_surfaces


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

    For higher dimensions the volume is computed using the determinant:
    .. math::
        V_{d} = \sqrt{\det{A A^T}}
    where .math:`A` is the array of vectors.

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


def compute_vectors(a: float, b: Optional[float] = None, c: Optional[float] = None,
                    alpha: Optional[float] = None, beta: Optional[float] = None,
                    gamma: Optional[float] = None) -> np.ndarray:
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
    return np.round(vectors, decimals=10)


class KDTree(cKDTree):
    """Simple wrapper of scipy's cKTree with global query settings."""

    def __init__(self, points, k=1, distance_bound=np.inf, eps=0., p=2):
        super().__init__(points)
        self.distance_bound = distance_bound
        self.k = k
        self.p = p
        self.eps = eps

    def query(self, x, n_jobs=1, k=0, eps=0., p=0, distance_upper_bound=0):
        k = k or self.k
        eps = eps or self.eps
        p = p or self.p
        bound = distance_upper_bound or self.distance_bound
        # noinspection PyUnresolvedReferences
        return super().query(x, k, eps, p, bound, n_jobs)


def compute_neighbours(positions, x=None, k=20, max_dist=np.inf, num_jobs=1, include_zero=False):
    x = positions if x is None else x
    tree = KDTree(positions, k=k, distance_bound=max_dist * 1.1, eps=0.1)
    distances, neighbours = tree.query(x, n_jobs=num_jobs)
    if not include_zero and np.all(distances[:, 0] == 0):
        distances = distances[:, 1:]
        neighbours = neighbours[:, 1:]
    invalid = distances > max_dist
    neighbours[invalid] = tree.n  # noqa
    distances[invalid] = np.inf
    return neighbours, distances


def create_lookup_table(array):
    values = np.sort(np.unique(array))
    indices = np.zeros_like(array, dtype=np.int8)
    for i, x in enumerate(values):
        mask = array == x
        indices[mask] = i
    return values, indices


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

    def draw(self, ax=None, color="C0", size=3, lw=1, alpha=0.15, point_color="k", point_size=3,
             draw_data=True, points=True, draw=True, fill=True):

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d" if self.dim == 3 else None)

        if draw_data:
            draw_points(ax, self.points, s=point_size**2, color=point_color)
            if self.dim > 1:
                draw_vectors(ax, self.points, lw=0.5, color=point_color)

        if points:
            draw_points(ax, self.vertices, s=size**2, color=color)

        if self.dim == 2 and draw:
            segments = np.array([self.vertices[i] for i in self.edges])
            draw_lines(ax, segments, color=color, lw=lw)
        elif self.dim == 3:
            if draw:
                segments = np.array([self.vertices[np.append(i, i[0])] for i in self.edges])
                draw_lines(ax, segments, color=color, lw=lw)
            if fill:
                surfaces = np.array([self.vertices[i] for i in self.edges])
                draw_surfaces(ax, surfaces, color=color, alpha=alpha)
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
        return np.array([np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)]).T

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
            raise ValueError("Either the number of points or the step size muste be specified")

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


def rotate2d(a, theta):
    """Applies the z-rotation matrix to a 2D point"""
    return np.dot(a, rz(theta)[:2, :2])


def rotate3d(a, thetax=0., thetay=0., thetaz=0.):
    """Applies the general rotation matrix to a 3D point"""
    return np.dot(a, rot(thetax, thetay, thetaz))
