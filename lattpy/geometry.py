# coding: utf-8
#
# This code is part of pylattice.
# 
# Copyright (c) 2020, Dylan Jones

import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, Voronoi
from .utils import distance
from .plotting import draw_line, draw_points, draw_vectors, draw_planes


class VoronoiTree:

    def __init__(self, points):
        points = np.asarray(points)
        dim = points.shape[1]
        edges = list()
        if dim > 1:
            vor = Voronoi(points)
            # Save only finite vertices
            vertices = vor.vertices  # noqa
            for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):  # noqa
                simplex = np.asarray(simplex)
                if np.all(simplex >= 0):
                    edges.append(simplex)
        else:
            vertices = points / 2
            idx = np.where((vertices == np.zeros(vertices.shape[1])).all(axis=1))[0]
            vertices = np.delete(vertices, idx)
            vertices = np.atleast_2d(vertices).T

        self.dim = dim
        self.points = points
        self.edges = edges
        self.vertices = vertices
        self.tree = cKDTree(points)  # noqa
        self.origin = self.query(np.zeros(dim))

    def query(self, x, k=1, eps=0):
        return self.tree.query(x, k, eps)  # noqa

    def draw(self, ax=None, color="C0", size=3, lw=1, alpha=0.15, point_color="k", point_size=3,
             draw_data=True, draw_vertices=True, draw_edges=True, draw_faces=True):

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d" if self.dim == 3 else None)

        if draw_data:
            draw_points(ax, self.points, s=point_size**2, color=point_color)
            if self.dim > 1:
                draw_vectors(ax, self.points, lw=0.5, color=point_color)
        if draw_vertices:
            draw_points(ax, self.vertices, s=size**2, color=color)

        draw_planes(ax, self.vertices, self.edges, color, color, lw, alpha, draw_edges, draw_faces)

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


def find_symmetry_points(cell):
    origin = np.zeros(cell.dim)
    corners = cell.points
    centers = list()
    for i, (p1, p2) in enumerate(cell.edges):
        center = p1 + (p2 - p1) / 2
        if not len(np.where(np.array(np.abs(cell.points - center) < 1e-5).all(axis=1))[0]):
            centers.append(center)
    centers = np.array(centers)
    lim = -0.01
    centers = centers[np.where(np.min(centers, axis=1) >= lim)[0]]
    corners = corners[np.where(corners[:, 0] >= lim)[0]]
    center = centers[np.argmax(centers[:, 0])]
    indices = np.argsort([distance(center, p) for p in corners])
    corners = corners[indices[:2]]
    return np.array([origin, center, *corners])


# =========================================================================
# OLD
# =========================================================================


def perp_vector(v: np.ndarray) -> np.ndarray:
    """Creates a perpendicular vector"""
    if len(v) == 2:
        return np.array([-v[1], v[0]])
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalizes an array."""
    return v / np.linalg.norm(v)


def normal_vector(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Computes a vector perpendicular to the two given vectors."""
    vec = np.cross(v1, v2)
    return vec / np.linalg.norm(vec)


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


def check_on_line(p0, v, point, epsilon=1e-10):
    """Checks if a point is on an infinite line."""
    t_check = (point - p0) / v
    t_check = t_check[np.isfinite(t_check)][0]
    return np.all(np.abs(p0 + t_check * v - point) < epsilon)


def check_on_linesegment(p0, v, point, epsilon=1e-6):
    """Checks if a point is on an line segment."""
    x = (point - p0) / v
    x = float(x[np.isfinite(x)][0])
    if not np.all(np.abs(p0 + x * v - point) < epsilon):
        return False
    if x is not None and 0 <= x <= +1:
        return True
    return False


def line_line_intersection(p1, v1, p2, v2, epsilon=1e-6):
    """Computes the intersection point of two infinite lines."""
    size = 1.0
    p1, v1 = p1 - size * v1, 2 * size * v1
    p2, v2 = p2 - size * v2, 2 * size * v2

    denom = np.linalg.norm(np.cross(v1, v2))
    if denom:
        s = np.linalg.norm(np.cross(v2, p2 - p1)) / denom
        inter1 = p1 + s * v1
        if check_on_line(p2, v2, inter1, epsilon):
            return inter1
        t = np.linalg.norm(np.cross(v1, p2 - p1)) / denom
        inter2 = p2 + t * v2
        if check_on_line(p1, v1, inter2, epsilon):
            return inter2
    return None


def line_plane_intersection(plane_point, normal_vec, line_point, line_vec, epsilon=1e-6):
    """Computes the intersection point of an infinite line and plane."""
    ndotu = normal_vec.dot(line_vec)
    if abs(ndotu) < epsilon:
        return None
    w = line_point - plane_point
    si = - normal_vec.dot(w) / ndotu
    return w + si * line_vec + plane_point


def plane_plane_intersection(p1, n1, p2, n2, epsilon=1e-6):
    """Computes the intersection line of two infinite planes."""
    v = np.cross(n1, n2)
    if np.sum(np.abs(v)) < epsilon:
        return None

    a = np.array([n1, n2, v])
    d = np.array([np.dot(p1, n1), np.dot(p2, n2), 0.]).reshape(3, 1)
    p0 = np.linalg.solve(a, d)[:, 0]

    return Line(p0, v / np.linalg.norm(v))


def line_segment_intersection(p0, v, point1, point2, epsilon=1e-6):
    """Computes the intersection point of an infinite line and a line segment."""
    p1, v1 = p0, v
    p2, v2 = point1, point2 - point1
    denom = np.linalg.norm(np.cross(v1, v2))
    if denom:
        s = np.linalg.norm(np.cross(v2, p2 - p1)) / denom
        t = np.linalg.norm(np.cross(v1, p1 - p2)) / denom
        inter1 = p1 + s * v1
        inter2 = p2 + t * v2
        if np.sum(np.abs(inter1 - inter2)) < epsilon:
            if check_on_linesegment(p2, v2, inter1, epsilon):
                return inter1
            if check_on_linesegment(p2, v2, inter2, epsilon):
                return inter2
    return None


class Line:

    def __init__(self, p0, v):
        self.p0 = np.asarray(p0)
        self.v = np.asarray(v)
        assert len(self.p0) == len(self.v)

    @classmethod
    def from_points(cls, b, a=None):
        if a is None:
            a = np.zeros_like(b)
        v = b - a
        return cls(a, v)

    @property
    def dim(self):
        return len(self.p0)

    @property
    def norm(self):
        return np.linalg.norm(self.v)

    def point(self, x):
        return self.p0 + x * self.v

    def factor(self, point, epsilon=1e-10):
        x = (point - self.p0) / self.v
        x = float(x[np.isfinite(x)][0])
        if not np.all(np.abs(self.p0 + x * self.v - point) < epsilon):
            return None
        return x

    def coeffs(self):
        n = perp_vector(self.v)
        d = np.dot(self.point(2), n)
        return np.append(n, d)

    def normalize(self):
        return self.__class__(self.p0, self.v / self.norm)

    def perp(self, p0=None):
        p0 = self.p0 if p0 is None else p0
        v_perp = perp_vector(self.v)
        return self.__class__(p0, v_perp)

    def intersection(self, other, epsilon=1e-6):
        if isinstance(other, Line):
            return line_line_intersection(self.p0, self.v, other.p0, other.v)
        elif isinstance(other, Plane):
            return line_plane_intersection(self.p0, self.v, other.p0, other.n, epsilon)
        return None

    def distance(self, points):
        return np.cross(self.v, points - self.p0) / np.linalg.norm(self.v)

    def __call__(self, x):
        return self.point(x)

    def data(self, x1, x0=0):
        return np.asarray([self.__call__(x0), self.__call__(x1)])

    def draw(self, ax, x1=1, x0=0, color=None):
        points = self.data(x1, x0)
        draw_line(ax, points, color=color)


class Plane:

    def __init__(self, p0, *args):
        if len(args) == 1:
            normal = args[0]
            v1 = perp_vector(normal)
            v2 = normal_vector(normal, v1)
        else:
            v1, v2 = args
        self.p0 = np.asarray(p0)
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2)
        assert len(self.p0) == len(self.v1)
        assert len(self.p0) == len(self.v2)

    @classmethod
    def from_points(cls, b, c, a=None):
        if a is None:
            a = np.zeros_like(b)
        v1 = b - a
        v2 = c - a
        return cls(a, v1, v2)

    @classmethod
    def from_normal(cls, p0, v):
        v1 = perp_vector(v)
        v2 = normal_vector(v, v1)
        return cls(p0, v1, v2)

    @property
    def dim(self):
        return len(self.p0)

    @property
    def n(self):
        return self.normal_vec()

    def coeffs(self):
        n = self.normal_vec()
        d = - np.dot(self.p0, n)
        return np.append(n, d)

    def normalize(self):
        v1 = self.v1 / np.linalg.norm(self.v1)
        v2 = self.v2 / np.linalg.norm(self.v2)
        return self.__class__(self.p0, v1, v2)

    def normal_vec(self):
        vec = np.cross(self.v1, self.v2)
        return vec / np.linalg.norm(vec)

    def normal_line(self):
        return Line(self.p0, self.normal_vec())

    def __call__(self, x1, x2):
        return self.p0 + x1 * self.v1 + x2 * self.v2

    def data(self, x1, x2, x10=0, x20=0):
        p1 = self.__call__(x10, x20)
        p2 = self.__call__(x1,  x20)
        p3 = self.__call__(x1,  x2)
        p4 = self.__call__(x10, x2)
        return np.asarray([p1, p2, p3, p4])

    def intersection(self, other, epsilon=1e-6):
        normal = self.normal_vec()
        if isinstance(other, Line):
            return line_plane_intersection(other.p0, other.v, self.p0, normal, epsilon)
        elif isinstance(other, Plane):
            return plane_plane_intersection(self.p0, normal, other.p0, other.normal_vec(), epsilon)
        return None
