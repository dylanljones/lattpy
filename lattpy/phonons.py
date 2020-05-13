# coding: utf-8
"""
Created on 13 May 2020
author: Dylan Jones
"""
import numpy as np
from .core.disptools import DispersionPath
import itertools


class DynamicMatrix:

    def __init__(self, n_base, dim):
        self.n_base = n_base
        self.dim = dim
        self.size = n_base * dim
        self.force_mats = dict()

        self._delta0 = np.zeros(self.dim)
        self._new_fc_mat(self._delta0)

    def _new_fc_mat(self, delta):
        self.force_mats[tuple(delta)] = np.zeros((self.size, self.size))

    def index(self, alpha, ax):
        return alpha * self.dim + ax

    def get_phi(self, delta=None):
        if delta is None:
            return self.get_phi(self._delta0)
        delta = tuple(delta)
        if delta not in self.force_mats.keys():
            self._new_fc_mat(delta)
        return self.force_mats[delta]

    def add_component(self, delta, phi, phi0=None):
        fc_mat = self.get_phi(delta)
        fc_mat += phi
        if phi0 is not None:
            fc_mat0 = self.get_phi()
            fc_mat0 += phi0

    def transform(self, q):
        dmat = np.zeros((self.size, self.size), dtype="complex")
        for delta, phi in self.force_mats.items():
            dmat += phi * np.exp(1j * np.dot(q, delta))
        return dmat

    def eigvals(self, q):
        dmat_q = self.transform(q)
        eigvals, eigvecs = np.linalg.eigh(dmat_q)
        return eigvals.real

    # =========================================================================

    def construct_phi(self, delta, alpha1, alpha2, const=1.0, mass1=1.0, mass2=1.0):
        v_delta = np.atleast_2d(delta)
        values = np.dot(v_delta.T, v_delta) / np.sum(np.square(v_delta))
        values *= const / np.sqrt(mass1 * mass2)

        phi0 = np.zeros((self.size, self.size))
        phi = np.zeros_like(phi0)
        for ax1, ax2 in itertools.product(range(self.dim), repeat=2):
            i, j = self.index(alpha1, ax1), self.index(alpha2, ax2)
            phi[i, j] = -values[ax1, ax2]
            if ax1 == ax2:
                phi0[i, j] = values[ax1, ax2]
        return phi, phi0

    def dispersion(self, q):
        q = np.atleast_1d(q) if self.dim == 1 else np.atleast_2d(q)
        omegas = np.zeros((len(q), self.dim * self.n_base))
        for i, qi in enumerate(q):
            omegas[i] = sorted(self.eigvals(qi))
        omegas[omegas < 0] = np.nan
        disp = np.sqrt(omegas)
        return disp[0] if len(q) == 1 else disp

    def bands(self, points, names=None, n=1000, cycle=True):
        if isinstance(points, DispersionPath):
            path = points
        else:
            path = DispersionPath(self.dim)
            path.add_points(points, names)
            if cycle:
                path.cycle()
        q = path.build(n)
        return self.dispersion(q)

    # =========================================================================

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
