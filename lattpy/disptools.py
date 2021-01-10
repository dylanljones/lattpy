# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Tools for dispersion computation and plotting."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from .utils import chain
from .spatial import distance
from .plotting import draw_lines
from .unitcell import Atom

__all__ = [
    "bandpath_subplots", "plot_dispersion", "disp_dos_subplots",
    "plot_disp_dos", "plot_bands", "DispersionPath"
]


def _color_list(color, num_bands):
    if color is None:
        colors = [f"C{i}" for i in range(num_bands)]
    elif isinstance(color, str) or not hasattr(color, "__len__"):
        colors = [color] * num_bands
    else:
        colors = color
    return colors


def _scale_xaxis(num_points, disp, scales=None):
    sect_size = len(disp) / (num_points - 1)
    scales = np.ones(num_points - 1) if scales is None else scales
    k0, k, ticks = 0, list(), [0]
    for scale in scales:
        k.extend(k0 + np.arange(sect_size) * scale)
        k0 = k[-1]
        ticks.append(k0)
    return k, ticks


def _set_piticks(axis, num_ticks=2, frmt=".1f"):
    axis.set_major_formatter(tck.FormatStrFormatter(rf'%{frmt} $\pi$'))
    axis.set_major_locator(tck.LinearLocator(2 * num_ticks + 1))


def bandpath_subplots(ticks, labels, xlabel="$k$", ylabel="$E(k)$", grid="both"):
    fig, ax = plt.subplots()
    ax.set_xlim(0, ticks[-1])
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        if not isinstance(grid, str):
            grid = "both"
        ax.set_axisbelow(True)
        ax.grid(b=True, which='major', axis=grid)
    return fig, ax


def _draw_dispersion(ax, k, disp, color=None, fill=False, alpha=0.2, lw=1.0):
    x = [0, np.max(k)]
    colors = _color_list(color, disp.shape[1])
    for i, band in enumerate(disp.T):
        col = colors[i]
        if isinstance(col, Atom):
            col = col.color
        ax.plot(k, band, lw=lw, color=col)
        if fill:
            ax.fill_between(x, min(band), max(band), color=col, alpha=alpha)


def plot_dispersion(disp, labels, xlabel="$k$", ylabel="$E(k)$", grid="both", color=None,
                    alpha=0.2, lw=1.0, scales=None, fill=False, ax=None, show=True):
    num_points = len(labels)
    k, ticks = _scale_xaxis(num_points, disp, scales)
    if ax is None:
        fig, ax = bandpath_subplots(ticks, labels, xlabel, ylabel, grid)
    else:
        fig = ax.get_figure()

    x = [0, np.max(k)]
    colors = _color_list(color, disp.shape[1])
    for i, band in enumerate(disp.T):
        col = colors[i]
        if isinstance(col, Atom):
            col = col.color
        ax.plot(k, band, lw=lw, color=col)
        if fill:
            ax.fill_between(x, min(band), max(band), color=col, alpha=alpha)

    fig.tight_layout()
    if show:
        plt.show()
    return ax


def disp_dos_subplots(ticks, labels, xlabel="$k$", ylabel="$E(k)$", doslabel="$n(E)$",
                      wratio=(3, 1), grid="both"):
    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": wratio}, sharey="all")
    ax1, ax2 = axs
    ax1.set_xlim(0, ticks[-1])
    if xlabel:
        ax1.set_xlabel(xlabel)
    if ylabel:
        ax1.set_ylabel(ylabel)
    if doslabel:
        ax2.set_xlabel(doslabel)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels)
    ax2.set_xticks([0])
    if grid:
        ax1.set_axisbelow(True)
        ax1.grid(b=True, which='major', axis=grid)
        ax2.set_axisbelow(True)
        ax2.grid(b=True, which='major', axis=grid)
    return fig, axs


def plot_disp_dos(disp, dos_data, labels, xlabel="k", ylabel="E(k)", doslabel="n(E)",
                  wratio=(3, 1), grid="both", color=None, fill=True, disp_alpha=0.2,
                  dos_alpha=0.2, lw=1.0, scales=None, axs=None, show=True):
    num_points = len(labels)
    k, ticks = _scale_xaxis(num_points, disp, scales)
    if axs is None:
        fig, axs = disp_dos_subplots(ticks, labels, xlabel, ylabel, doslabel, wratio, grid)
        ax1, ax2 = axs
    else:
        ax1, ax2 = axs
        fig = ax1.get_figure()

    x = [0, np.max(k)]
    colors = _color_list(color, disp.shape[1])
    for i, band in enumerate(disp.T):
        col = colors[i]
        if isinstance(col, Atom):
            col = col.color
        ax1.plot(k, band, lw=lw, color=col)
        if fill:
            ax1.fill_between(x, min(band), max(band), color=col, alpha=disp_alpha)

    for i, band in enumerate(dos_data):
        col = colors[i]
        if isinstance(col, Atom):
            col = col.color
        bins, dos = band
        ax2.plot(dos, bins, lw=lw, color=col)
        ax2.fill_betweenx(bins, 0, dos, alpha=dos_alpha, color=col)

    ax2.set_xlim(0, ax2.get_xlim()[1])
    fig.tight_layout()
    if show:
        plt.show()
    return axs


def plot_bands(kgrid, bands, k_label="k", disp_label="E(k)", grid="both", contour_grid=False,
               bz=None, pi_ticks=True, ax=None, show=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    dim = len(bands.shape) - 1
    if dim == 1:
        k = kgrid[0]
        ax.plot(k, bands.T)
        if k_label:
            ax.set_xlabel(f"${k_label}$")
        if disp_label:
            ax.set_ylabel(f"${disp_label}$")
        if grid:
            ax.grid(axis=grid)

        if pi_ticks:
            _set_piticks(ax.xaxis, num_ticks=2)
        ax.set_xlim(np.min(k), np.max(k))

        if bz is not None:
            for x in bz:
                ax.axvline(x=x, color="k")

    elif dim == 2:
        kx, ky = kgrid
        kxx, kyy = np.meshgrid(kx, ky)
        if len(bands) == 1:
            bands = bands[0]
        else:
            bands = np.sum(np.abs(bands), axis=0)

        im = ax.contourf(kxx, kyy, bands)
        ax.set_aspect('equal')
        if k_label:
            ax.set_xlabel(f"{k_label}$_x$")
            ax.set_ylabel(f"{k_label}$_y$")
        if disp_label:
            label = ""
            if disp_label:
                label = disp_label if len(bands) == 1 else f"|{disp_label}|"
            fig.colorbar(im, ax=ax, label=label)
        if grid and contour_grid:
            ax.grid(axis=grid)

        if pi_ticks:
            _set_piticks(ax.xaxis, num_ticks=2)
            _set_piticks(ax.yaxis, num_ticks=2)

        if bz is not None:
            draw_lines(ax, bz, color="k")
    else:
        raise NotImplementedError()

    fig.tight_layout()
    if show:
        plt.show()
    return ax


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
    labels: list of str
    points: list of array_like
    n_sect: int
    """

    def __init__(self, dim=0):
        self.dim = dim
        self.labels = list()
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
        self.labels.append(name)
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
        self.labels.append(self.labels[0])
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
        return np.arange(self.num_points) * self.n_sect, self.labels

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

    def subplots(self, xlabel="k", ylabel="E(k)", grid="both"):
        """ Creates an empty matplotlib plot with configured axes for the path.

        Parameters
        ----------
        xlabel: str, optional
        ylabel: str, optional
        grid: str, optional

        Returns
        -------
        fig: plt.Figure
        ax: plt.Axis
        """
        ticks, labels = self.get_ticks()
        return bandpath_subplots(ticks, labels, xlabel, ylabel, grid)

    def plot_dispersion(self, disp, ax=None, show=True, **kwargs):
        scales = self.scales()
        return plot_dispersion(disp, self.labels, scales=scales, ax=ax, show=show,  **kwargs)

    def plot_disp_dos(self, disp, dos, axs=None, show=True, **kwargs):
        scales = self.scales()
        return plot_disp_dos(disp, dos, self.labels, scales=scales, axs=axs, show=show, **kwargs)
