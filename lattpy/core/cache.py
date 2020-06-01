# coding: utf-8
"""
Created on 15 May 2020
author: Dylan Jones
"""
import numpy as np
import copy


class LatticeCache:

    def __init__(self, dim):
        self.dim = dim
        self.indices = None
        self.positions = None
        self.neighbours = None
        self.periodic_neighbours = None

    @property
    def n(self):
        return len(self.indices) if self.indices is not None else 0

    def copy(self):
        data = LatticeCache(self.dim)
        if self.indices is not None:
            indices = self.indices.copy()
            positions = self.positions.copy()
            neighbours = copy.deepcopy(self.neighbours)
            data.set(indices, positions, neighbours)
        if self.periodic_neighbours is not None:
            data.set_periodic_neighbours(copy.deepcopy(self.periodic_neighbours))
        return data

    def reset(self):
        self.indices = None
        self.neighbours = None
        self.periodic_neighbours = None

    def set(self, indices, positions, neighbours):
        self.indices = np.asarray(indices)
        self.positions = np.asarray(positions)
        self.neighbours = neighbours

    def set_periodic_neighbours(self, neighbours):
        self.periodic_neighbours = neighbours

    def get_nvec(self, i):
        return self.indices[i, :-1]

    def get_alpha(self, i):
        return self.indices[i, -1]

    def get_index(self, i):
        index = self.indices[i]
        return index[:-1], index[-1]

    def get_neighbours(self, i, dist=0):
        neighbours = list(self.neighbours[i][dist])
        if self.periodic_neighbours is not None:
            neighbours += list(self.periodic_neighbours[i][dist])
        return neighbours

    def get_limits(self):
        return np.array([np.min(self.positions, axis=0), np.max(self.positions, axis=0)])

    def site_mask(self, mins=(), maxs=(), invert=False):
        if len(mins) != self.dim:
            mins = list(mins) + [None] * (self.dim - len(mins))
        if len(maxs) != self.dim:
            maxs = list(maxs) + [None] * (self.dim - len(maxs))
        mins = [(x if x is not None else -np.inf) for x in mins]
        maxs = [(x if x is not None else +np.inf) for x in maxs]
        limits = np.array([mins, maxs])
        mask = 1
        for ax in range(self.dim):
            ax_data = self.positions[:, ax]
            mask &= (limits[0, ax] <= ax_data) & (ax_data <= limits[1, ax])

        return 1 - mask if invert else mask

    def find_sites(self, mins=(), maxs=(), invert=False):
        mask = self.site_mask(mins, maxs, invert)
        return np.where(mask)[0]

    def find_outer_sites(self, ax, offset):
        limits = self.get_limits()

        mins = [None for _ in range(self.dim)]
        maxs = [None for _ in range(self.dim)]
        mins[ax] = limits[0, ax] + offset
        maxs[ax] = limits[1, ax] - offset
        mask = self.site_mask(mins, maxs, invert=True)
        return np.where(mask)[0]

    def __bool__(self):
        return self.indices is not None
