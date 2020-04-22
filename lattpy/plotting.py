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


class LatticePlot:

    def __init__(self, fig=None, ax=None, size=10, color=True, lw=1., dim3=False):
        self.dim3 = dim3
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d" if dim3 else None)
        self.fig, self.ax = fig, ax

        self.radius = size
        self.atom_size = size ** 2
        self.color = color
        self.lw = lw

    @classmethod
    def subplots(cls, nrows=1, ncols=1, **kwargs):
        fig, ax = plt.subplots(nrows, ncols, **kwargs)
        return cls(fig, ax)

    @property
    def dpi(self):
        return self.fig.dpi

    @property
    def size(self):
        return self.fig.get_size_inches() * self.dpi

    def get_xlim(self):
        return self.ax.get_xlim()

    def get_ylim(self):
        return self.ax.get_ylim()

    def get_xdatalim(self):
        bbox = self.ax.dataLim
        return bbox.x0, bbox.x1

    def get_ydatalim(self):
        bbox = self.ax.dataLim
        return bbox.y0, bbox.y1

    def get_zdatalim(self):
        bbox = self.ax.dataLim
        return bbox.z0, bbox.z1

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

    def autoscale(self, autoscale=True, axis='both', tight=None):
        self.ax.autoscale(autoscale, axis, tight)

    def set_margins(self, *margin):
        if len(margin) == 1:
            margin = np.ones(3 if self.dim3 else 2) * margin[0]

        if self.dim3:
            self.ax.margins(x=margin[0], y=margin[1], z=margin[2])
        else:
            self.ax.margins(x=margin[0], y=margin[1])

    def set_padding(self, *padding):
        if len(padding) == 1:
            padding = np.ones(3 if self.dim3 else 2) * padding[0]

        x0, x1 = self.get_xdatalim()
        self.ax.set_xlim(x0 - padding[0], x1 + padding[0])
        y0, y1 = self.get_ydatalim()
        self.ax.set_ylim(y0 - padding[1], y1 + padding[1])

        if self.dim3:
            z0, z1 = self.get_zdatalim()
            self.ax.set_zlim(z0 - padding[2], z1 + padding[2])

    def set_limits(self, x=None, y=None, z=None):
        if x is not None:
            self.ax.set_xlim(*x)
        if y is not None:
            self.ax.set_ylim(*y)
        if z is not None:
            self.ax.set_ylim(*z)

    def set_labels(self, x='', y='', z=''):
        if x:
            self.ax.set_xlabel(x)
        if y:
            self.ax.set_ylabel(y)
        if z:
            self.ax.set_zlabel(z)

    # =========================================================================

    def draw_linecollection(self, segments, *args, **kwargs):
        if self.dim3:
            coll = Line3DCollection(segments, *args, **kwargs)
        else:
            coll = LineCollection(segments, *args, **kwargs)
        self.ax.add_collection(coll)

    def draw_vector(self, vector, pos=None, lw=1, ls='-', **kwargs):
        pos = pos if pos is not None else np.zeros(3 if self.dim3 else 2)
        segments = [[pos, pos + vector]]
        self.draw_linecollection(segments, linestyles=ls, linewidths=lw, **kwargs)

    def draw_vectors(self, vectors, pos=None, lw=1, ls='-', **kwargs):
        pos = pos if pos is not None else np.zeros(3 if self.dim3 else 2)
        vectors = np.asarray(vectors).T
        segments = list()
        for v in vectors:
            segments.append([pos, pos + v])
        self.draw_linecollection(segments, linestyles=ls, linewidths=lw, **kwargs)

    def print_indices(self, positions, offset=0.1):
        offset = np.ones_like(positions[0]) * offset
        for i, pos in enumerate(positions):
            lowerleft = np.asarray(pos) + offset
            self.ax.text(*lowerleft, s=str(i), va="bottom", ha="left")

    def draw_line(self, p1, p2, *args, **kwargs):
        self.draw_linecollection([[p1, p2]], *args, **kwargs)

    def draw_sites(self, atom, positions):
        positions = np.asarray(positions)
        col = atom.col
        size = atom.size**2
        label = atom.label()
        scat = self.ax.scatter(*positions.T, zorder=3, s=size, color=col, label=label, alpha=1)

        # self.ax.ignore_existing_data_limits = True
        datalim = scat.get_datalim(self.ax.transData)
        self.ax.update_datalim(datalim)

        # limits = np.array([np.min(points, axis=0), np.max(points, axis=0)])
        # if self.limits is None:
        #     self.limits = limits
        # else:
        #     self.limits[0, self.limits[0] > limits[0]] = limits[0, self.limits[0] > limits[0]]
        #     self.limits[1, self.limits[1] < limits[1]] = limits[1, self.limits[1] < limits[1]]

    # =========================================================================

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

    def legend(self, *args, **kwargs):
        self.ax.legend(*args, **kwargs)

    def tight(self, *args, **kwargs):
        self.fig.tight_layout(*args, **kwargs)

    def grid(self, **kwargs):
        self.ax.grid(**kwargs)

    def set_view(self, azim=30, angle=0):
        if self.dim3:
            self.ax.view_init(azim, angle)

    def setup(self):
        if self.dim3:
            self.set_view(30, 30)
        else:
            self.set_equal_aspect()
            self.ax.grid()
        # self.legend()

    @staticmethod
    def draw(sleep=1e-10):
        plt.draw()
        plt.pause(sleep)

    def show(self, enabled=True, tight=True):
        if tight:
            self.tight()
        if enabled:
            plt.show()
