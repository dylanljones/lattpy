# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Objects for representing the shape of a finite lattice."""

import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import itertools
from abc import ABC, abstractmethod
from .plotting import draw_lines, draw_surfaces


class AbstractShape(ABC):
    """Abstract shape object."""

    def __init__(self, dim, pos=None):
        self.dim = dim
        self.pos = np.zeros(dim) if pos is None else np.array(pos)

    @abstractmethod
    def limits(self):
        """Returns the limits of the shape."""
        pass

    @abstractmethod
    def contains(self, points, tol=0.):
        """Checks if the given points are contained in the shape."""
        pass

    @abstractmethod
    def plot(self, ax, color="k", lw=0.0, alpha=0.2, **kwargs):
        """Plots the contour of the shape."""
        pass

    def __repr__(self):
        return self.__class__.__name__


# noinspection PyUnresolvedReferences,PyShadowingNames
class Shape(AbstractShape):
    """General shape object.

    Examples
    --------

    Cartesian coordinates

    >>> points = np.random.uniform(-0.5, 2.5, size=(500, 2))
    >>> s = lp.Shape((2, 2))
    >>> s.limits()
    [[0.  2. ]
     [0.  2.]]

    >>> import matplotlib.pyplot as plt
    >>> mask = s.contains(points)
    >>> s.plot(plt.gca())
    >>> plt.scatter(*points[mask].T, s=3, color="g")
    >>> plt.scatter(*points[~mask].T, s=3, color="r")
    >>> plt.gca().set_aspect("equal")
    >>> plt.show()

    Angled coordinate system

    >>> s = lp.Shape((2, 2), basis=[[1, 0.2], [0, 1]])
    >>> s.limits()
    [[0.  2. ]
     [0.  2.4]]

    >>> mask = s.contains(points)
    >>> s.plot(plt.gca())
    >>> plt.scatter(*points[mask].T, s=3, color="g")
    >>> plt.scatter(*points[~mask].T, s=3, color="r")
    >>> plt.gca().set_aspect("equal")
    >>> plt.show()
    """

    def __init__(self, shape, pos=None, basis=None):
        if not hasattr(shape, "__len__"):
            shape = [shape]
        super().__init__(len(shape), pos)
        self.size = np.array(shape)
        self.basis = None if basis is None else np.array(basis)

    def _build(self):
        corners = list(itertools.product(*zip(np.zeros(self.dim), self.size)))
        corners = self.pos + np.array(corners)
        edges = None
        surfs = None
        if self.dim == 2:
            edges = np.array([[0, 1], [0, 2], [2, 3], [3, 1]])
            surfs = np.array([[0, 1, 3, 2, 0]])
        elif self.dim == 3:
            # Edge indices
            edges = np.array([
                [0, 2], [2, 3], [3, 1], [1, 0],
                [4, 6], [6, 7], [7, 5], [5, 4],
                [0, 4], [2, 6], [3, 7], [1, 5]
            ])
            # Surface indices
            surfs = np.array([
                [0, 2, 3, 1],
                [4, 6, 7, 5],
                [0, 4, 6, 2],
                [2, 6, 7, 3],
                [3, 7, 5, 1],
                [1, 5, 4, 0]
            ])
        if self.basis is not None:
            corners = np.inner(corners, self.basis.T)
        return corners, edges, surfs

    def limits(self):
        corners, _, _ = self._build()
        lims = np.array([np.min(corners, axis=0), np.max(corners, axis=0)])
        return lims.T

    def contains(self, points, tol=0.):
        if self.basis is not None:
            points = np.inner(points, np.linalg.inv(self.basis.T))
        mask = np.logical_and(self.pos - tol <= points,
                              points <= self.pos + self.size + tol)
        return np.all(mask, axis=1)

    def plot(self, ax, color="k", lw=0.0, alpha=0.2, **kwargs):  # pragma: no cover
        corners, edges, surfs = self._build()
        segments = corners[edges]
        lines = draw_lines(ax, segments, color=color, lw=lw)
        segments = corners[surfs]
        if self.dim < 3:
            surfaces = ax.fill(*segments.T, color=color, alpha=alpha)
        elif self.dim == 3:
            surfaces = draw_surfaces(ax, segments, color=color, alpha=alpha)
        else:
            raise NotImplementedError("Can't plot shape in D>3!")
        return lines, surfaces


# noinspection PyUnresolvedReferences,PyShadowingNames
class Circle(AbstractShape):
    """Circle shape.

    Examples
    --------

    >>> s = lp.Circle((0, 0), radius=2)
    >>> s.limits()
    [[-2.  2.]
     [-2.  2.]]

    >>> import matplotlib.pyplot as plt
    >>> points = np.random.uniform(-2, 2, size=(500, 2))
    >>> mask = s.contains(points)
    >>> s.plot(plt.gca())
    >>> plt.scatter(*points[mask].T, s=3, color="g")
    >>> plt.scatter(*points[~mask].T, s=3, color="r")
    >>> plt.gca().set_aspect("equal")
    >>> plt.show()

    """

    def __init__(self, pos, radius):
        super().__init__(len(pos), pos)
        self.radius = radius

    def limits(self):
        rad = np.full(self.dim, self.radius)
        lims = self.pos + np.array([-rad, +rad])
        return lims.T

    def contains(self, points, tol=0.):
        dists = np.sqrt(np.sum(np.square(points - self.pos), axis=1))
        return dists <= self.radius + tol

    def plot(self, ax, color="k", lw=0.0, alpha=0.2, **kwargs):  # pragma: no cover
        xy = tuple(self.pos)
        line = plt.Circle(xy, self.radius, lw=lw, color=color, fill=False)
        ax.add_artist(line)
        surf = plt.Circle(xy, self.radius, lw=0, color=color, alpha=alpha, fill=True)
        ax.add_artist(surf)
        return line, surf


# noinspection PyUnresolvedReferences,PyShadowingNames
class Donut(AbstractShape):
    """Circle shape with cut-out in the middle.

    Examples
    --------

    >>> s = lp.Donut((0, 0), radius_outer=2, radius_inner=1)
    >>> s.limits()
    [[-2.  2.]
     [-2.  2.]]

    >>> import matplotlib.pyplot as plt
    >>> points = np.random.uniform(-2, 2, size=(500, 2))
    >>> mask = s.contains(points)
    >>> s.plot(plt.gca())
    >>> plt.scatter(*points[mask].T, s=3, color="g")
    >>> plt.scatter(*points[~mask].T, s=3, color="r")
    >>> plt.gca().set_aspect("equal")
    >>> plt.show()

    """

    def __init__(self, pos, radius_outer, radius_inner):
        super().__init__(len(pos), pos)
        self.radii = np.array([radius_inner, radius_outer])

    def limits(self):
        rad = np.full(self.dim, self.radii[1])
        lims = self.pos + np.array([-rad, +rad])
        return lims.T

    def contains(self, points, tol=1e-10):
        dists = np.sqrt(np.sum(np.square(points - self.pos), axis=1))
        return np.logical_and(self.radii[0] - tol <= dists,
                              dists <= self.radii[1] + tol)

    def plot(self, ax, color="k", lw=0.0, alpha=0.2, **kwargs):  # pragma: no cover
        n = 100

        theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
        xs = np.outer(self.radii, np.cos(theta))
        ys = np.outer(self.radii, np.sin(theta))
        # in order to have a closed area, the circles
        # should be traversed in opposite directions
        xs[1, :] = xs[1, ::-1]
        ys[1, :] = ys[1, ::-1]

        line1 = ax.plot(xs[0], ys[0], color=color, lw=lw)[0]
        line2 = ax.plot(xs[1], ys[1], color=color, lw=lw)[0]
        surf = ax.fill(np.ravel(xs), np.ravel(ys), fc=color, alpha=alpha, ec=None)

        return [line1, line2], surf


# noinspection PyUnresolvedReferences,PyShadowingNames
class ConvexHull(AbstractShape):
    """Shape defined by convex hull of arbitrary points.

    Examples
    --------

    >>> s = lp.ConvexHull([[0, 0], [2, 0], [2, 1], [1, 2], [0, 2]])
    >>> s.limits()
    [[0.  2.]
     [0.  2.]]

    >>> import matplotlib.pyplot as plt
    >>> points = np.random.uniform(-0.5, 2.5, size=(500, 2))
    >>> mask = s.contains(points)
    >>> s.plot(plt.gca())
    >>> plt.scatter(*points[mask].T, s=3, color="g")
    >>> plt.scatter(*points[~mask].T, s=3, color="r")
    >>> plt.gca().set_aspect("equal")
    >>> plt.show()

    """

    def __init__(self, points):
        dim = len(points[0])
        super().__init__(dim)
        self.hull = scipy.spatial.ConvexHull(points)

    def limits(self):
        points = self.hull.points
        return np.array([np.min(points, axis=0), np.max(points, axis=0)]).T

    def contains(self, points, tol=1e-10):
        return np.all(np.add(np.dot(points, self.hull.equations[:, :-1].T),
                             self.hull.equations[:, -1]) <= tol, axis=1)

    def plot(self, ax, color="k", lw=0.0, alpha=0.2, **kwargs):  # pragma: no cover

        if self.dim == 2:
            segments = self.hull.points[self.hull.simplices]
            lines = draw_lines(ax, segments, color=color, lw=lw)
            # segments = self.hull.points[surf]
            segments = self.hull.points[self.hull.vertices]
            surfaces = ax.fill(*segments.T, fc=color, alpha=alpha, ec=None)

        elif self.dim == 3:

            segments = np.array(
                [self.hull.points[np.append(i, i[0])] for i in self.hull.simplices]
            )
            lines = draw_lines(ax, segments, color=color, lw=lw)

            surfaces = np.array([self.hull.points[i] for i in self.hull.simplices])
            draw_surfaces(ax, surfaces, color=color, alpha=alpha)
        else:
            raise NotImplementedError("Can't plot shape in D>3!")

        return lines, surfaces
