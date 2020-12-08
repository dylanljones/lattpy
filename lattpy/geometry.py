# coding: utf-8
#
# This code is part of pylattice.
# 
# Copyright (c) 2020, Dylan Jones

import numpy as np
from .utils import distance


def perpendicular_vector(v):
    if len(v) == 2:
        return np.array([-v[1], v[0]])
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])


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
        return len(self.v)

    @property
    def norm(self):
        return np.linalg.norm(self.v)

    def normalize(self):
        return self.__class__(self.p0, self.v / self.norm)

    def perp(self, p0=None):
        p0 = self.p0 if p0 is None else p0
        v_perp = perpendicular_vector(self.v)
        return self.__class__(p0, v_perp)

    def intersection(self, other):
        p1, v1 = self.p0, self.v
        if isinstance(other, Line):
            p2, v2 = other.p0, other.v
        else:
            p2, v2 = other
        inter = p2 + np.linalg.norm(np.cross(v1, p1 - p2)) / np.linalg.norm(np.cross(v1, v2)) * v2
        if np.isnan(inter).any() or np.isinf(inter).any():
            return None
        return inter

    def __call__(self, x):
        return self.p0 + x * self.v

    def data(self, x1, x0=0):
        return np.asarray([self.__call__(x0), self.__call__(x1)]).T


def get_perp_line(pos, pos0=None):
    line = Line.from_points(pos, pos0)
    return line.perp(line(0.5))


def line_intersections(lines):
    pairs = list()
    intersections = list()
    for i, line1 in enumerate(lines):
        for line2 in lines[:]:
            inter = line1.intersection(line2)
            if inter is not None:
                pairs.append((line1, line2))
                intersections.append(inter)
    return np.unique(intersections, axis=0, return_index=False)


def _minimize_point_distances(points):
    results = list()
    results.append(points.pop(0))
    while points:
        distances = [distance(results[-1], p) for p in points]
        i_min = np.argmin(distances)
        p = points.pop(i_min)
        if not np.isclose(p, np.asarray(results)).all(axis=1).any():
            results.append(p)
    return results


def wigner_seitz_points(neighbour_cell_positions):
    lines = [get_perp_line(pos) for pos in neighbour_cell_positions]
    intersections = line_intersections(lines)
    inters = list()
    for i, point in enumerate(intersections):
        if not np.isclose(point, neighbour_cell_positions).all(axis=1).any():
            inters.append(point)
    return np.array(_minimize_point_distances(inters))


def wigner_seitz_limits(latt):
    points = wigner_seitz_points(latt)
    return np.array([np.min(points, axis=0), np.max(points, axis=0)]).T


def wigner_seitz_arange(latt, steps, offset=0.):
    limits = wigner_seitz_limits(latt) * (1 + offset)
    steps = [steps] * latt.dim if not hasattr(steps, "__len__") else steps
    return [np.arange(*lims, step=step) for lims, step in zip(limits, steps)]


def wigner_seitz_linspace(latt, nums, offset=0.):
    limits = wigner_seitz_limits(latt) * (1 + offset)
    nums = [nums] * latt.dim if not hasattr(nums, "__len__") else nums
    return [np.linspace(*lims, num=num) for lims, num in zip(limits, nums)]


def wigner_seitz_mesh(latt, nums=None, steps=None, offset=0.):
    if nums is not None:
        return np.array(np.meshgrid(*wigner_seitz_linspace(latt, nums, offset=offset)))
    elif steps is not None:
        return np.array(np.meshgrid(*wigner_seitz_arange(latt, steps, offset=offset)))


class WignerSeitzCell:

    def __init__(self, points):
        self.points = np.asarray(points)
        self._limits = np.array([np.min(points, axis=0), np.max(points, axis=0)]).T

    @property
    def limits(self):
        return self._limits

    @property
    def size(self):
        return self._limits[1] - self._limits[0]

    @property
    def dim(self):
        return len(self._limits)

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
            return np.array(np.meshgrid(*self.linspace(nums, offset)))
        elif steps is not None:
            return np.array(np.meshgrid(*self.arange(steps, offset)))
        return None


def wigner_seitz_cell(latt):
    nvecs = list(latt.get_neighbour_cells())
    positions = [latt.translate(nvec) for nvec in nvecs]
    intersections = wigner_seitz_points(positions)
    return WignerSeitzCell(intersections)
