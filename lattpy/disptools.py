# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
import matplotlib.pyplot as plt
from .utils import chain
from .spatial import distance
from .plotting import draw_lines


def band_subplots(ticks, labels, x_label="k", disp_label="E(k)", grid="both"):
    fig, ax = plt.subplots()
    ax.set_xlim(0, ticks[-1])
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    if x_label:
        ax.set_xlabel(f"${x_label}$")
    if disp_label:
        ax.set_ylabel(f"${disp_label}$")

    if grid:
        ax.grid(axis=grid)
    fig.tight_layout()
    return fig, ax


def plot_dispersion(ax, disp, fill=True, alpha=0.2, lw=1.0):
    for band in disp.T:
        ax.plot(band, lw=lw)
        if fill:
            ax.fill_between([0, len(band)], min(band), max(band), alpha=alpha)


def plot_bands(disp, labels, x_label="k", disp_label="E(k)", grid="both",
               fill=True, alpha=0.2, lw=1.0, show=True):
    num_points = len(labels)
    ticks = np.arange(num_points) * len(disp) / (num_points - 1)

    fig, ax = band_subplots(ticks, labels, x_label, disp_label, grid)
    plot_dispersion(ax, disp, fill, alpha, lw)

    if show:
        plt.show()
    return fig, ax


def band_dos_subplots(ticks, labels, x_label="k", disp_label="E(k)",
                      dos_label="n(E)", wratio=(3, 1), grid="both"):
    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": wratio}, sharey="all")
    ax1, ax2 = axs

    ax1.set_xlim(0, ticks[-1])
    if x_label:
        ax1.set_xlabel(f"${x_label}$")
    if disp_label:
        ax1.set_ylabel(f"${disp_label}$")
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels)

    if dos_label:
        ax2.set_xlabel(f"${dos_label}$")
    ax2.set_xticks([0])

    if grid:
        ax1.grid(axis=grid)
        ax2.grid(axis=grid)
    fig.tight_layout()
    return fig, axs


def plot_band_dos(disp, bins, dos, labels, x_label="k", disp_label="E(k)", dos_label="n(E)",
                  wratio=(3, 1), grid="both", fill=True, disp_alpha=0.2, dos_color="C0",
                  dos_alpha=0.2, lw=1.0, show=True):
    num_points = len(labels)
    ticks = np.arange(num_points) * len(disp) / (num_points - 1)

    fig, axs = band_dos_subplots(ticks, labels, x_label, disp_label, dos_label, wratio, grid)
    ax1, ax2 = axs
    plot_dispersion(ax1, disp, fill=fill, alpha=disp_alpha, lw=lw)
    ax2.plot(dos, bins, lw=lw, color=dos_color)
    ax2.fill_betweenx(bins, 0, dos, alpha=dos_alpha, color=dos_color)
    ax2.set_xlim(0, ax2.get_xlim()[1])

    if show:
        plt.show()
    return fig, axs


class DispersionPath:
    """ This object is used to define a dispersion path between high symmetry (HS) points.

    Examples
    --------
    Define a path using the add-method or preset points. To get the actual points the
    'build'-method is called:
    >>> path = DispersionPath(dim=3).add([0, 0, 0], 'Gamma').x(a=1.0).cycle()
    >>> vectors = path.build(n_sect=1000)

    Attributes
    ----------
    dim: int
    names: list of str
    points: list of array_like
    n_sect: int
    """

    def __init__(self, dim=0):
        self.dim = dim
        self.names = list()
        self.points = list()
        self.n_sect = 0

    @classmethod
    def chain_path(cls, a=1.0):
        return cls(dim=1).x(a).gamma().cycle()

    @classmethod
    def square_path(cls, a=1.0):
        return cls(dim=2).gamma().x(a).m(a).cycle()

    @classmethod
    def cubic_path(cls, a=1.0):
        return cls(dim=3).gamma().x(a).m(a).gamma().r(a)

    @property
    def num_points(self):
        """ int: Number of HS points in the path"""
        return len(self.points)

    def add(self, point, name=""):
        """ Adds a new HS point to the path

        This method returns the instance for easier path definitions.

        Parameters
        ----------
        point: array_like
            The coordinates of the HS point. If the dimension of the point is
            higher than the set dimension the point will be clipped.
        name: str, optional
            Optional name of the point. If not specified the number of the point is used.

        Returns
        -------
        self: DispersionPath
        """
        if not name:
            name = str(len(self.points))
        point = np.asarray(point)
        if self.dim:
            point = point[:self.dim]
        else:
            self.dim = len(point)
        self.points.append(point)
        self.names.append(name)
        return self

    def add_points(self, points, names=None):
        """ Adds multiple HS points to the path

        Parameters
        ----------
        points: array_like
            The coordinates of the HS points.
        names: list of str, optional
            Optional names of the points. If not specified the number of the point is used.

        Returns
        -------
        self: DispersionPath
        """
        if names is None:
            names = [""] * len(points)
        for point, name in zip(points, names):
            self.add(point, name)
        return self

    def cycle(self):
        """ Adds the first point of the path.

        This method returns the instance for easier path definitions.

        Returns
        -------
        self: DispersionPath
        """
        self.points.append(self.points[0])
        self.names.append(self.names[0])
        return self

    def gamma(self):
        r""" DispersionPath: Adds the .math:'\Gamma=(0, 0, 0)' point to the path """
        return self.add([0, 0, 0], r"$\Gamma$")

    def x(self, a=1.0):
        r""" DispersionPath: Adds the .math:'X=(\pi, 0, 0)' point to the path """
        return self.add([np.pi / a, 0, 0], r"$X$")

    def m(self, a=1.0):
        r""" DispersionPath: Adds the ,math:'M=(\pi, \pi, 0)' point to the path """
        return self.add([np.pi / a, np.pi / a, 0], r"$M$")

    def r(self, a=1.0):
        r""" DispersionPath: Adds the .math:'R=(\pi, \pi, \pi)' point to the path """
        return self.add([np.pi / a, np.pi / a, np.pi / a], r"$R$")

    def build(self, n_sect=1000):
        """ Builds the vectors defining the path between the set HS points.

        Parameters
        ----------
        n_sect: int, optional
            Number of points between each pair of HS points.

        Returns
        -------
        path: (N, D) np.ndarray
        """
        self.n_sect = n_sect
        path = np.zeros((0, self.dim))
        for p0, p1 in chain(self.points):
            path = np.append(path, np.linspace(p0, p1, num=n_sect), axis=0)
        return path

    def get_ticks(self):
        """ Get the positions of the points of the last buildt path.

        Mainly used for setting ticks in plot.

        Returns
        -------
        ticks: (N) np.ndarray
        labels: (N) list
        """
        return np.arange(self.num_points) * self.n_sect, self.names

    def edges(self):
        """Constructs the edges of the path."""
        return list(chain(self.points))

    def distances(self):
        """Computes the distances between the edges of the path."""
        dists = list()
        for p0, p1 in self.edges():
            dists.append(distance(p0, p1))
        return np.array(dists)

    def scales(self):
        """Computes the scales of the the edges of the path."""
        dists = self.distances()
        return dists / dists[0]

    def draw(self, ax, color=None, lw=1., **kwargs):
        lines = draw_lines(ax, self.edges(), color=color, lw=lw, **kwargs)
        return lines

    def subplots(self, xlabel="", ylabel="", grid=True):
        """ Creates an empty matplotlib plot with configured axes for the path.

        Parameters
        ----------
        xlabel: str, optional
        ylabel: str, optional
        grid: bool, optional

        Returns
        -------
        fig: plt.Figure
        ax: plt.Axis
        """
        ticks, labels = self.get_ticks()
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, ticks[-1])
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        if grid:
            ax.grid()
        return fig, ax
