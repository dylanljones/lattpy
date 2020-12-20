# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import itertools
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Line3D, Poly3DCollection

# Golden ratio as standard ratio for plot-figures
GOLDEN_RATIO = (np.sqrt(5) - 1.0) / 2.0

# =========================================================================
# Formatting
# =========================================================================


def _pts_to_inch(pts):
    return pts * (1. / 72.27)


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
    vectors = np.atleast_2d(vectors).T
    dim = len(vectors)
    pos = pos if pos is not None else np.zeros(dim)
    if dim != 3:
        kwargs.update({"angles": "xy", "scale_units": "xy", "scale": 1})
    return ax.quiver(*pos, *vectors, **kwargs)


def draw_vectors(ax, vectors, pos=None, ls="-", lw=1, **kwargs):
    pos = pos if pos is not None else np.zeros(len(vectors[0]))
    vectors = np.atleast_2d(vectors)
    segments = list()
    for v in vectors:
        segments.append([pos, pos + v])
    return draw_lines(ax, segments, linestyles=ls, linewidths=lw, **kwargs)


def draw_points(ax, points, s=10, color=None, alpha=1.0, **kwargs):
    # Fix 1D case
    points = np.atleast_1d(points)
    if len(points.shape) == 1:
        points = points[:, np.newaxis]
    if points.shape[1] == 1:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))

    scat = ax.scatter(*points.T, s=s, color=color, alpha=alpha, **kwargs)
    # Manualy update data-limits
    # ax.ignore_existing_data_limits = True
    datalim = scat.get_datalim(ax.transData)
    ax.update_datalim(datalim)
    return scat


def draw_sites(ax, positions, size, **kwargs):
    positions = np.asarray(positions)
    scat = ax.scatter(*positions.T, zorder=3, s=size**2, **kwargs)
    # Manualy update data-limits
    ax.ignore_existing_data_limits = True
    datalim = scat.get_datalim(ax.transData)
    ax.update_datalim(datalim)
    return scat


def draw_indices(ax, positions, offset=0.1):
    offset = np.ones_like(positions[0]) * offset
    texts = list()
    for i, pos in enumerate(positions):
        lowerleft = np.asarray(pos) + offset
        txt = ax.text(*lowerleft, s=str(i), va="bottom", ha="left")
        texts.append(txt)
    return texts


def draw_cell(ax, vectors, color="k", lw=2, outlines=True):
    dim = len(vectors)
    if dim == 1:
        draw_arrows(ax, [vectors[0, 0], 0], color=color, lw=lw)
        return

    draw_arrows(ax, vectors, color=color, lw=lw)
    if outlines:
        for v, pos in itertools.permutations(vectors, r=2):
            data = np.asarray([pos, pos + v]).T
            ax.plot(*data, color=color, ls='--', lw=1)
        if dim == 3:
            for vecs in itertools.permutations(vectors, r=3):
                v, pos = vecs[0], np.sum(vecs[1:], axis=0)
                data = np.asarray([pos, pos + v]).T
                ax.plot(*data, color=color, ls='--', lw=1)


def draw_planes(ax, vertices, edges, lc=None, fc=None, lw=1, alpha=0.15, draw=True, fill=True):
    try:
        dim = len(vertices[0])
    except TypeError:
        dim = 1
    if dim == 2 and draw:
        segments = np.array([vertices[i] for i in edges])
        draw_lines(ax, segments, color=lc, lw=lw)
    elif dim == 3:
        if draw:
            segments = np.array([vertices[np.append(i, i[0])] for i in edges])
            draw_lines(ax, segments, color=lc, lw=lw)
        if fill:
            for simplex in edges:
                verts = [list(vertices[simplex])]
                poly = Poly3DCollection(verts, alpha=alpha, facecolor=fc, linewidths=1)
                ax.add_collection3d(poly)
