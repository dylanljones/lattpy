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
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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


def draw_arrows(ax, vectors, pos=None, **kwargs):
    vectors = np.atleast_2d(vectors).T
    dim = len(vectors)
    pos = pos if pos is not None else np.zeros(dim)
    if dim != 3:
        kwargs.update({"angles": "xy", "scale_units": "xy", "scale": 1})
    return ax.quiver(*pos, *vectors, **kwargs)


def draw_vectors(ax, vectors, pos=None, ls="-", lw=1, **kwargs):
    pos = pos if pos is not None else np.zeros(2)
    vectors = np.atleast_2d(vectors)
    segments = list()
    for v in vectors:
        segments.append([pos, pos + v])
    return draw_lines(ax, segments, linestyles=ls, linewidths=lw, **kwargs)


def draw_sites(ax, positions, size, **kwargs):
    positions = np.asarray(positions)
    scat = ax.scatter(*positions.T, zorder=3, s=size**2, alpha=1, **kwargs)
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