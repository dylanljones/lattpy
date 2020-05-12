# coding: utf-8
"""
Created on 12 Apr 2020
author: Dylan Jones
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Golden ratio as standard ratio for plot-figures
GOLDEN_RATIO = (np.sqrt(5) - 1.0) / 2.0


def _pts_to_inch(pts):
    return pts * (1. / 72.27)


class Plot:

    def __init__(self, dim3=False, fig=None, ax=None, **kwargs):
        self.dim3 = dim3
        if fig is None:
            fig = plt.figure(**kwargs)
            ax = fig.add_subplot(111, projection="3d" if dim3 else None)
        self.fig, self.ax = fig, ax

    @property
    def dpi(self):
        return self.fig.dpi

    @property
    def size(self):
        return self.fig.get_size_inches() * self.dpi

    @property
    def xlim(self):
        return self.ax.get_xlim()

    @property
    def ylim(self):
        return self.ax.get_ylim()

    @property
    def zlim(self):
        return self.ax.get_zlim()

    @property
    def xdatalim(self):
        bbox = self.ax.dataLim
        return bbox.x0, bbox.x1

    @property
    def ydatalim(self):
        bbox = self.ax.dataLim
        return bbox.y0, bbox.y1

    @property
    def zdatalim(self):
        bbox = self.ax.dataLim
        return bbox.z0, bbox.z1

    def update_datalim(self, datalim):
        self.ax.update_datalim(datalim)

    def set_figsize(self, width=None, height=None, ratio=None):
        ratio = ratio or GOLDEN_RATIO
        # Width and height
        if (width is not None) and (height is not None):
            width = _pts_to_inch(width)
            height = _pts_to_inch(height)
        elif (width is not None) and (height is None):
            width = _pts_to_inch(width)
            height = width * ratio
        elif (width is None) and (height is not None):
            height = _pts_to_inch(height)
            width = height / ratio
        else:
            raise ValueError('Not enough inputs!')
        self.fig.set_size_inches(width, height)

    def set_equal_aspect(self):
        self.ax.set_aspect("equal", "box")

    def set_scales(self, x=None, y=None, z=None):
        if x is not None:
            self.ax.set_xscale(x)
        if y is not None:
            self.ax.set_yscale(y)
        if z is not None:
            self.ax.set_zscale(z)

    def set_margins(self, x=None, y=None, z=None):
        if z is None:
            self.ax.margins(x=x, y=y)
        else:
            self.ax.margins(x=x, y=y, z=z)

    def set_padding(self, x=None, y=None, z=None):
        if x is not None:
            if not hasattr(x, '__len__'):
                x = (x, x)
            x0, x1 = self.xdatalim
            self.ax.set_xlim(x0 - x[0], x1 + x[1])
        if y is not None:
            if not hasattr(x, '__len__'):
                y = (y, y)
            y0, y1 = self.ydatalim
            self.ax.set_ylim(y0 - y[0], y1 + y[1])
        if z is not None:
            if not hasattr(x, '__len__'):
                z = (z, z)
            z0, z1 = self.zdatalim
            self.ax.set_ylim(z0 - z[0], z1 + z[1])

    def set_limits(self, x=None, y=None, z=None):
        if x is not None:
            if hasattr(x, "__len__"):
                lim = self.ax.get_xlim()
                lim0 = lim[0] if x[0] is None else x[0]
                lim1 = lim[1] if x[1] is None else x[1]
            else:
                lim0, lim1 = -x, +x
            self.ax.set_xlim(lim0, lim1)
        if y is not None:
            if hasattr(y, "__len__"):
                lim = self.ax.get_ylim()
                lim0 = lim[0] if y[0] is None else y[0]
                lim1 = lim[1] if y[1] is None else y[1]
            else:
                lim0, lim1 = -y, +y
            self.ax.set_ylim(lim0, lim1)
        if z is not None:
            if hasattr(z, "__len__"):
                lim = self.ax.get_zlim()
                lim0 = lim[0] if y[0] is None else y[0]
                lim1 = lim[1] if y[1] is None else y[1]
            else:
                lim0, lim1 = -z, +z
            self.ax.set_zlim(lim0, lim1)

    def set_labels(self, x=None, y=None, z=None):
        if x is not None:
            self.ax.set_xlabel(x)
        if y is not None:
            self.ax.set_ylabel(y)
        if z is not None:
            self.ax.set_zlabel(z)

    def set_xticks(self, ticks, labels=None, minor=False, fontdict=None, **kwargs):
        self.ax.set_xticks(ticks, minor)
        if labels is not None:
            self.ax.set_xticklabels(labels, fontdict, minor, **kwargs)

    def set_yticks(self, ticks, labels=None, minor=False, fontdict=None, **kwargs):
        self.ax.set_yticks(ticks, minor)
        if labels is not None:
            self.ax.set_yticklabels(labels, fontdict, minor, **kwargs)

    def set_zticks(self, ticks, labels=None, minor=False, fontdict=None, **kwargs):
        self.ax.set_zticks(ticks, minor)
        if labels is not None:
            self.ax.set_zticklabels(labels, fontdict, minor, **kwargs)

    def colorbar(self, im, *args, orientation="vertical", **kwargs):
        divider = make_axes_locatable(self.ax)
        if orientation == "vertical":
            cax = divider.append_axes("right", size="5%", pad=0.05)
        elif orientation == "horizontal":
            cax = divider.append_axes("bottom", size="5%", pad=0.6)
        else:
            allowed = ["vertical", "horizontal"]
            raise ValueError(f"Invalid orientation: {orientation}. Must be in {allowed}")
        return self.fig.colorbar(im, ax=self.ax, cax=cax, orientation=orientation, *args, **kwargs)

    def get_legend_handles(self):
        return self.ax.get_legend_handles_labels()[0]

    def legend(self, loc=None, handles=None, add_handles=None, *args, **kwargs):
        if handles is None:
            handles = self.get_legend_handles()
            if add_handles is not None:
                handles += add_handles
        self.ax.legend(loc=loc, handles=handles, *args, **kwargs)

    def grid(self, below_axis=True, **kwargs):
        self.ax.set_axisbelow(below_axis)
        self.ax.grid(**kwargs)

    def set_view(self, azim=30, angle=0):
        if self.dim3:
            self.ax.view_init(azim, angle)

    def tight(self, *args, **kwargs):
        self.fig.tight_layout(*args, **kwargs)

    def save(self, *relpaths, dpi=600, rasterized=True):
        import os

        if rasterized:
            self.ax.set_rasterized(True)
        file = os.path.join(*relpaths)
        self.fig.savefig(file, dpi=dpi)
        print(f"Figure {file} saved")

    def setup(self, **kwargs):
        self.set_scales(x=kwargs.get('xscale', None),
                        y=kwargs.get('yscale', None),
                        z=kwargs.get('zscale', None))
        self.set_margins(x=kwargs.get('xmargin', None),
                         y=kwargs.get('ymargin', None),
                         z=kwargs.get('zmargin', None))
        self.set_padding(x=kwargs.get('xpadding', None),
                         y=kwargs.get('ypadding', None),
                         z=kwargs.get('zpadding', None))
        self.set_limits(x=kwargs.get('xlim', None),
                        y=kwargs.get('ylim', None),
                        z=kwargs.get('zlim', None))
        self.set_labels(x=kwargs.get('xlabel', None),
                        y=kwargs.get('ylabel', None),
                        z=kwargs.get('zlabel', None))
        if 'xticks' in kwargs:
            labels = kwargs.get('xticklabels', None)
            self.set_xticks(kwargs['xticks'], labels)
        if 'yticks' in kwargs:
            labels = kwargs.get('yticklabels', None)
            self.set_yticks(kwargs['yticks'], labels)
        if 'zticks' in kwargs:
            labels = kwargs.get('zticklabels', None)
            self.set_zticks(kwargs['zticks'], labels)

        if kwargs.get('grid', False):
            self.grid()
        if kwargs.get('legend', False):
            self.legend()
        if kwargs.get('tight', False):
            self.tight()

    def show(self, tight=True, enabled=True, **kwargs):
        if kwargs:
            self.setup(**kwargs)
        if tight:
            self.tight()
        if enabled:
            plt.show()

    # =========================================================================
    # Drawing
    # =========================================================================

    def text(self, pos, s, va="center", ha="center", *args, **kwargs):
        self.ax.text(*pos, s=s, va=va, ha=ha, *args, **kwargs)

    def fill(self, x, y1, y2=0, alpha=0.25, invert_axis=False, *args, **kwargs):
        if invert_axis:
            return self.ax.fill_betweenx(x, y1, y2, alpha=alpha, *args, **kwargs)
        else:
            return self.ax.fill_between(x, y1, y2, alpha=alpha, *args, **kwargs)

    def draw_lines(self, x=None, y=None, *args, **kwargs):
        lines = list()
        if x is not None:
            for _x in np.atleast_1d(x):
                lines.append(self.ax.axvline(_x, *args, **kwargs))
        if y is not None:
            for _y in np.atleast_1d(y):
                lines.append(self.ax.axhline(_y, *args, **kwargs))
        return lines

    def draw_linecollection(self, segments, *args, **kwargs):
        if self.dim3:
            coll = Line3DCollection(segments, *args, **kwargs)
        else:
            coll = LineCollection(segments, *args, **kwargs)
        self.ax.add_collection(coll)

    def draw_vectors(self, vectors, pos=None, lw=1, ls='-', **kwargs):
        pos = pos if pos is not None else np.zeros(2)
        vectors = np.atleast_2d(vectors).T
        segments = list()
        for v in vectors:
            segments.append([pos, pos + v])
        self.draw_linecollection(segments, linestyles=ls, linewidths=lw, **kwargs)

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot(self, *args, **kwargs):
        return self.ax.plot(*args, **kwargs)[0]

    def plotfill(self, x, y, alpha=0.25, **kwargs):
        line = self.ax.plot(x, y, **kwargs)[0]
        self.fill(x, y, color=line.get_color(), alpha=alpha)
        return line

    def scatter(self, x, y, s=None, *args, **kwargs):
        s = None if s is None else s**2
        return self.ax.scatter(x, y, s=s, *args, **kwargs)


# =========================================================================


class LatticePlot(Plot):

    def __init__(self, fig=None, ax=None, dim3=False):
        super().__init__(dim3, fig, ax)

    def draw_vector(self, vector, pos=None, lw=1, ls='-', **kwargs):
        pos = pos if pos is not None else np.zeros(3 if self.dim3 else 2)
        segments = [[pos, pos + vector]]
        self.draw_linecollection(segments, linestyles=ls, linewidths=lw, **kwargs)

    def print_indices(self, positions, offset=0.1):
        offset = np.ones_like(positions[0]) * offset
        for i, pos in enumerate(positions):
            lowerleft = np.asarray(pos) + offset
            self.ax.text(*lowerleft, s=str(i), va="bottom", ha="left")

    def draw_sites(self, atom, positions):
        positions = np.asarray(positions)
        col = atom.col
        size = atom.size**2
        label = atom.label()
        scat = self.ax.scatter(*positions.T, zorder=3, s=size, color=col, label=label, alpha=1)

        # self.ax.ignore_existing_data_limits = True
        datalim = scat.get_datalim(self.ax.transData)
        self.ax.update_datalim(datalim)
