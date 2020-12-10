# coding: utf-8
#
# This code is part of pylattice.
# 
# Copyright (c) 2020, Dylan Jones

import numpy as np
import itertools
from .utils import distance
from .plotting import draw_line, draw_lines, draw_plane


def perp_vector(v):
    if len(v) == 2:
        return np.array([-v[1], v[0]])
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])


def normalize(v):
    return v / np.linalg.norm(v)


def normal_vector(v1, v2):
    vec = np.cross(v1, v2)
    return vec / np.linalg.norm(vec)


def rx(theta):
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])


def ry(theta):
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, +cos]])


def rz(theta):
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])


def rot(thetax=0., thetay=0., thetaz=0.):
    r = np.eye(3)
    if thetaz:
        r = np.dot(r, rz(thetaz))
    if thetay:
        r = np.dot(r, ry(thetay))
    if thetax:
        r = np.dot(r, rz(thetax))
    return r


def rotate2d(a, theta):
    return np.dot(a, rz(theta)[:2, :2])


def rotate3d(a, thetax=0., thetay=0., thetaz=0.):
    return np.dot(a, rot(thetax, thetay, thetaz))


def _check_on_line(p0, v, point, epsilon=1e-10):
    t_check = (point - p0) / v
    t_check = t_check[np.isfinite(t_check)][0]
    return np.all(np.abs(p0 + t_check * v - point) < epsilon)


def line_line_intersection(p1, v1, p2, v2, epsilon=1e-6):
    size = 1.0
    p1, v1 = p1 - size * v1, 2 * size * v1
    p2, v2 = p2 - size * v2, 2 * size * v2

    denom = np.linalg.norm(np.cross(v1, v2))
    if denom:
        s = np.linalg.norm(np.cross(v2, p2 - p1)) / denom
        inter1 = p1 + s * v1
        if _check_on_line(p2, v2, inter1, epsilon):
            return inter1
        t = np.linalg.norm(np.cross(v1, p2 - p1)) / denom
        inter2 = p2 + t * v2
        if _check_on_line(p1, v1, inter2, epsilon):
            return inter2
    return None


def line_plane_intersection(plane_point, normal_vec, line_point, line_vec, epsilon=1e-6):
    ndotu = normal_vec.dot(line_vec)
    if abs(ndotu) < epsilon:
        return None
    w = line_point - plane_point
    si = - normal_vec.dot(w) / ndotu
    return w + si * line_vec + plane_point


def plane_plane_intersection(p1, n1, p2, n2, epsilon=1e-6):
    v = np.cross(n1, n2)
    if np.sum(np.abs(v)) < epsilon:
        return None

    a = np.array([n1, n2, v])
    d = np.array([np.dot(p1, n1), np.dot(p2, n2), 0.]).reshape(3, 1)
    p0 = np.linalg.solve(a, d)[:, 0]

    return Line(p0, v / np.linalg.norm(v))


# ==================================================================================================


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

    def __call__(self, x):
        return self.p0 + x * self.v

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

    @property
    def hesse(self):
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

    def draw(self, ax, x1=1, x2=1, x10=0, x20=0, size=3, color=None, draw=True, fill=True,
             alpha=0.2, lw=0.5):
        points = self.data(x1, x2, x10, x20)
        draw_plane(ax, points, size, color, draw, fill, alpha, lw)

    def intersection(self, other, epsilon=1e-6):
        normal = self.normal_vec()
        if isinstance(other, Line):
            return line_plane_intersection(other.p0, other.v, self.p0, normal, epsilon)
        elif isinstance(other, Plane):
            return plane_plane_intersection(self.p0, normal, other.p0, other.normal_vec(), epsilon)
        return None


# =========================================================================


def get_intersections(items):
    intersections = list()
    for item1, item2 in itertools.permutations(items, r=2):
        inter = item1.intersection(item2)
        if inter is not None:
            intersections.append(inter)
    return intersections


def unique(a, axis=0, tol=1e-10):
    scale = 1 / tol
    return np.unique(np.floor(scale*a) / scale, axis=axis)


def wigner_seitz_points(positions):
    if len(positions[0]) > 2:
        planes = list()
        for vec in positions:
            vec = np.array(vec)
            plane = Plane(vec / 2, vec)
            planes.append(plane)
        lines = get_intersections(planes)
    else:
        lines = list()
        for pos in positions:
            line = Line.from_points(pos)
            pline = line.perp(line(0.5))
            lines.append(pline)

    # Remove lattice points from intersections
    points = list()
    for p in get_intersections(lines):
        if not len(np.where(np.array(np.abs(positions - p) < 1e-5).all(axis=1))[0]):
            points.append(p)

    points = np.array(points)
    return unique(points, axis=0)


def check_point2d(points, point, origin):
    point_distance = distance(point, origin)
    dists = [distance(point, p) for p in points]
    if point_distance > max(dists):
        return False

    p0, p1 = points[np.argsort(dists)[:2]]
    x1, v1 = p0, p1 - p0
    x2, v2 = point, origin - point
    denom = np.linalg.norm(np.cross(v1, v2))
    if denom == 0:
        return True

    s = np.linalg.norm(np.cross(v2, x2 - x1)) / denom
    inter = np.asarray(x1 + s * v1)
    inter_dist = distance(inter, origin)
    return point_distance < inter_dist


def find_edges(points):
    points = np.asarray(points)
    num_points, dim = points.shape
    all_indices = np.arange(num_points)
    pairs = list()
    for i, p1 in enumerate(points):
        indices = np.delete(all_indices, i)
        distances = np.array([distance(p1, points[j]) for j in indices])
        neighbours = indices[np.argsort(distances)[:dim]]
        pairs.extend([sorted([i, j]) for j in neighbours])
    return np.unique(pairs, axis=0)


class WignerSeitzCell:

    def __init__(self, points):
        self.points = np.array([])
        self.edges = np.array([])
        self.verts = np.array([])
        self.limits = np.array([])
        self.init(points)

    def init(self, points):
        self.points = points
        self.limits = np.array([np.min(self.points, axis=0), np.max(self.points, axis=0)]).T
        self.edges = np.array([])
        self.verts = np.array([])
        dim = len(self.limits)
        if dim > 1:
            self.edges = np.asarray(find_edges(self.points))
        if dim > 2:
            pass

    @property
    def size(self):
        return self.limits[1] - self.limits[0]

    @property
    def dim(self):
        return len(self.limits)

    def edge_lines(self):
        lines = list()
        for i, j in self.edges:
            lines.append(self.points[[i, j]])
        return lines

    def arange(self, steps, offset=0.):
        limits = self.limits * (1 + offset)
        steps = [steps] * self.dim if not hasattr(steps, "__len__") else steps
        return [np.arange(*lims, step=step) for lims, step in zip(limits, steps)]

    def linspace(self, nums, offset=0.):
        limits = self.limits * (1 + offset)
        nums = [nums] * self.dim if not hasattr(nums, "__len__") else nums
        return [np.linspace(*lims, num=num) for lims, num in zip(limits, nums)]

    def meshgrid(self, nums=None, steps=None, offset=0.):
        if nums is not None:
            grid = np.array(np.meshgrid(*self.linspace(nums, offset)))
        elif steps is not None:
            grid = np.array(np.meshgrid(*self.arange(steps, offset)))
        else:
            raise ValueError("Either the number of points or the step size muste be specified")

        return grid

    def draw(self, ax, color=None, lw=1.0, **kwargs):
        draw_lines(ax, self.edge_lines(), color=color, lw=lw, **kwargs)


def find_symmetry_points(cell):
    origin = np.zeros(cell.dim)
    corners = cell.points
    centers = list()
    for i, item in enumerate(cell.edges):
        p1, p2 = cell.points[item]
        line = Line.from_points(p1, p2)
        center = line(0.5)
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
