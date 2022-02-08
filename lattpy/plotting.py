# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Contains plotting tools for the lattice and other related objects."""

import itertools
import numpy as np
from collections.abc import Iterable
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection, Collection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Line3D, Poly3DCollection
from matplotlib.artist import allow_rasterization
from matplotlib import path, transforms

__all__ = [
    "set_margins", "set_padding", "set_limits", "draw_line", "draw_lines",
    "draw_arrows", "draw_vectors", "draw_points", "draw_indices", "draw_cell",
    "draw_surfaces", "interpolate_to_grid", "draw_sites"
]

# Golden ratio as standard ratio for plot-figures
GOLDEN_RATIO = (np.sqrt(5) - 1.0) / 2.0

# =========================================================================
# Formatting
# =========================================================================


def set_margins(ax, x=None, y=None, z=None):
    if z is None:
        ax.margins(x=x, y=y)
    else:
        ax.margins(x=x, y=y, z=z)


def set_padding(ax, x=None, y=None, z=None):
    bbox = ax.dataLim
    if x is not None:
        if not hasattr(x, '__len__'):
            x = (x, x)
        ax.set_xlim(bbox.x0 - x[0], bbox.x1 + x[1])
    if y is not None:
        if not hasattr(y, '__len__'):
            y = (y, y)
        ax.set_ylim(bbox.x0 - y[0], bbox.y1 + y[1])
    if z is not None:
        if not hasattr(z, '__len__'):
            z = (z, z)
        ax.set_zlim(bbox.z0 - z[0], bbox.z0 + z[1])


def set_limits(ax, dim, padding=None, margins=0.1):
    if padding is not None:
        if not hasattr(padding, "__len__"):
            padding = [padding] * dim
        set_padding(ax, *padding)
    else:
        if not hasattr(margins, "__len__"):
            margins = [margins] * dim
        set_margins(ax, *margins)


# =========================================================================
# Plotting
# =========================================================================


def subplot(dim, adjustable="datalim", ax=None):
    if dim > 3:
        raise ValueError(f"Plotting in {dim} dimensions is not supported!")
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)
    else:
        fig = ax.get_figure()
    if dim < 3:
        ax.set_aspect("equal", adjustable)
    return fig, ax


def draw_lines(ax, segments, *args, **kwargs):
    dim = len(segments[0][0])
    if dim == 3:
        coll = Line3DCollection(segments, *args, **kwargs)
    else:
        coll = LineCollection(segments, *args, **kwargs)
    ax.add_collection(coll)
    return coll


def draw_line(ax, points, *args, **kwargs):
    dim = len(points[0])
    if dim < 3:
        line = Line2D(*points.T, *args, **kwargs)
        ax.add_line(line)
    elif dim == 3:
        line = Line3D(*points.T, *args, **kwargs)
        ax.add_line(line)
    else:
        raise ValueError(f"Can't plot lines with dimension {dim}")
    return line


def draw_arrows(ax, vectors, pos=None, **kwargs):
    vectors = np.atleast_2d(vectors)
    num_vecs, dim = vectors.shape
    if pos is None:
        pos = np.zeros((num_vecs, dim))
    else:
        pos = np.asarray(pos)
        if len(pos) == 1 or pos.shape[0] == 1:
            pos = np.tile(pos, (num_vecs, 1))
    assert len(pos) == len(vectors)

    points = pos.T
    directions = vectors.T
    if dim != 3:
        kwargs.update({"angles": "xy", "scale_units": "xy", "scale": 1})
    else:
        kwargs.update({"normalize": False})
    return ax.quiver(*points, *directions, **kwargs)


def draw_vectors(ax, vectors, pos=None, ls="-", lw=1, zorder=1, **kwargs):
    if not len(vectors):
        return None
    pos = pos if pos is not None else np.zeros(len(vectors[0]))
    vectors = np.atleast_2d(vectors)
    # Fix 1D case
    if vectors.shape[1] == 1:
        vectors = np.hstack((vectors, np.zeros((vectors.shape[0], 1))))
        pos = np.array([pos[0], 0])
    segments = list()
    for v in vectors:
        segments.append([pos, pos + v])
    return draw_lines(ax, segments, linestyles=ls, linewidths=lw, zorder=zorder,
                      **kwargs)


def draw_points(ax, points, size=10, color=None, alpha=1.0, zorder=3, **kwargs):
    points = np.atleast_2d(points)
    # Fix 1D case
    if points.shape[1] == 1:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))

    scat = ax.scatter(*points.T, s=size**2, color=color, alpha=alpha, zorder=zorder,
                      **kwargs)
    # Manualy update data-limits
    # ax.ignore_existing_data_limits = True
    datalim = scat.get_datalim(ax.transData)
    ax.update_datalim(datalim)
    return scat


# noinspection PyAbstractClass
class CircleCollection(Collection):
    """Custom circle collection

    The default matplotlib `CircleCollection` creates circles based on their
    area in screen units. This class uses the radius in data units. It behaves
    like a much faster version of a `PatchCollection` of `Circle`.
    The implementation is similar to `EllipseCollection`.
    """
    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)
        self.radius = np.atleast_1d(radius)
        self._paths = [path.Path.unit_circle()]
        self.set_transform(transforms.IdentityTransform())
        self._transforms = np.empty((0, 3, 3))

    def _set_transforms(self):
        ax = self.axes
        self._transforms = np.zeros((self.radius.size, 3, 3))
        self._transforms[:, 0, 0] = self.radius * ax.bbox.width / ax.viewLim.width
        self._transforms[:, 1, 1] = self.radius * ax.bbox.height / ax.viewLim.height
        self._transforms[:, 2, 2] = 1

    @allow_rasterization
    def draw(self, renderer):
        self._set_transforms()
        super().draw(renderer)


def draw_sites(ax, points, radius=10, color=None, alpha=1.0, zorder=3, **kwargs):
    points = np.atleast_2d(points)
    # Fix 1D case
    if points.shape[1] == 1:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))

    dim = points.shape[1]
    if dim < 3:
        col = CircleCollection(radius, offsets=points, transOffset=ax.transData,
                               color=color, alpha=alpha, zorder=zorder, **kwargs)
        ax.add_collection(col)
        label = kwargs.get("label", "")
        if label:
            ax.plot([], [], marker='o', lw=0, color=color, label=label, markersize=10)
        datalim = col.get_datalim(ax.transData)
        ax.update_datalim(datalim)
        return col
    else:
        size = radius * 50
        scat = ax.scatter(*points.T, s=size**2, color=color, alpha=alpha, zorder=zorder,
                          **kwargs)
        # Manualy update data-limits
        # ax.ignore_existing_data_limits = True
        datalim = scat.get_datalim(ax.transData)
        ax.update_datalim(datalim)
        return scat


def draw_indices(ax, positions, offset=0.1):
    offset = np.ones_like(positions[0]) * offset
    texts = list()
    for i, pos in enumerate(positions):
        if len(pos) == 1:
            pos = [pos, 0]
        lowerleft = np.asarray(pos) + offset
        txt = ax.text(*lowerleft, s=str(i), va="bottom", ha="left")
        texts.append(txt)
    return texts


def draw_cell(ax, vectors, color="k", lw=2., zorder=1, outlines=True):
    dim = len(vectors)
    if dim == 1:
        draw_arrows(ax, [vectors[0, 0], 0], color=color, lw=lw, zorder=zorder)
        return

    draw_arrows(ax, vectors, color=color, lw=lw, zorder=zorder)
    # draw_vectors(ax, vectors, color=color, lw=lw, zorder=zorder)
    if outlines:
        for v, pos in itertools.permutations(vectors, r=2):
            data = np.asarray([pos, pos + v]).T
            ax.plot(*data, color=color, ls='--', lw=1, zorder=zorder)
        if dim == 3:
            for vecs in itertools.permutations(vectors, r=3):
                v, pos = vecs[0], np.sum(vecs[1:], axis=0)
                data = np.asarray([pos, pos + v]).T
                ax.plot(*data, color=color, ls='--', lw=1, zorder=zorder)


def draw_surfaces(ax, vertices, color=None, alpha=0.5):
    if not isinstance(vertices[0][0], Iterable):
        vertices = [list(vertices)]
    poly = Poly3DCollection(vertices, alpha=alpha, facecolor=color)
    ax.add_collection3d(poly)
    return poly


def interpolate_to_grid(positions, values, num=(100, 100), offset=(0., 0.),
                        method="linear", fill_value=np.nan):
    x, y = positions.T

    # Create regular grid
    xi = np.linspace(min(x) - offset[0], max(x) + offset[0], num[0])
    yi = np.linspace(min(y) - offset[1], max(y) + offset[1], num[1])

    # Interpolate data to grid
    xx, yy = np.meshgrid(xi, yi)
    zz = griddata((x, y), values, (xi[None, :], yi[:, None]), method, fill_value)
    return xx, yy, zz
