# coding: utf-8
"""
Created on 10 May 2020
author: Dylan Jones
"""
import numpy as np
import matplotlib.pyplot as plt
from .core.vector import vlinspace, chain


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

    def __init__(self, dim=3):
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
        self.points.append(point[:self.dim])
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
            path = np.append(path, vlinspace(p0, p1, n_sect), axis=0)
        return path

    def get_ticks(self):
        """ Get the positions of the points of the last buildt path.

        Mainly used for setting ticks in plot.

        Returns
        -------
        ticks: (N) np.ndarray
        """
        return np.arange(self.num_points) * self.n_sect, self.names

    def new_plot(self, xlabel="", ylabel="", grid=True):
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

    def bands(self, disp_func, n_sect=1000, **kwargs):
        """ Wrapper for calling the given dispersion-method for the whole path.

        Parameters
        ----------
        disp_func: method
            A method for computing the dispersion for one point. This method is called
            for all points of the path.
        n_sect: int, optional
            Number of points between each pair of HS points used when building the path.
        **kwargs
            Optional keyword arguments of the dispersion method.

        Returns
        -------
        bands: np.ndarray
        """
        q_vecs = self.build(n_sect)
        n = len(q_vecs)
        omega = disp_func(q_vecs[0], **kwargs)
        bands = np.zeros((n, len(omega)))
        bands[0] = omega
        for i in range(1, n):
            bands[i] = disp_func(q_vecs[i], **kwargs)
        return bands


# =========================================================================
# PHONONS
# =========================================================================


class DynamicMatrix:

    def __init__(self, n_base, dim):
        self.n_base = n_base
        self.dim = dim
        self.size = n_base * dim
        self.force_mats = dict()

    def __repr__(self):
        string = self.__class__.__name__
        string += f"(base: {self.n_base}, {self.dim}D)"
        return string

    def frmt_fc_matrices(self):
        width = 2 + max([len(str(k)) for k in self.force_mats.keys()])
        parts = list()
        for key, array in self.force_mats.items():
            header = f"{key}:"
            mlines = str(array).splitlines()
            mlines[-1] = mlines[-1][:-1]
            string = f"{header:<{width}}" + str(mlines[0][1:])
            for row in mlines[1:]:
                string += "\n" + (" " * width) + row[1:]
            parts.append(string)
        return "\n\n".join(parts)

    def __str__(self):
        string = self.__repr__() + "\n"
        string += "-" * len(string) + "\n"
        return string + self.frmt_fc_matrices()

    def index(self, alpha, ax):
        return alpha * self.dim + ax

    def get_fc_mat(self, key):
        key = tuple(key)
        if key not in self.force_mats.keys():
            self.force_mats[key] = np.zeros((self.size, self.size))
        return self.force_mats[key]

    def add_fc(self, key, item, value):
        fc = self.get_fc_mat(key)
        fc[item] += value

    def __call__(self, delta):
        return self.force_mats[tuple(delta)]

    def transform(self, q):
        dmat = np.zeros((self.size, self.size), dtype="complex")
        for delta, phi in self.force_mats.items():
            dmat += phi * np.exp(1j * np.dot(q, delta))
        return dmat

    def eigvals(self, q):
        dmat_q = self.transform(q)
        eigvals, eigvecs = np.linalg.eigh(dmat_q)
        return eigvals.real

    def dispersion(self, q):
        q = np.atleast_1d(q) if self.dim == 1 else np.atleast_2d(q)
        omegas = np.zeros((len(q), self.dim * self.n_base))
        for i, qi in enumerate(q):
            omegas[i] = sorted(self.eigvals(qi))
        omegas[omegas < 0] = np.nan
        disp = np.sqrt(omegas)
        return disp[0] if len(q) == 1 else disp
