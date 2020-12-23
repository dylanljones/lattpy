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
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, Voronoi
from .utils import chain
from .plotting import draw_points, draw_vectors, draw_lines, draw_surfaces


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
