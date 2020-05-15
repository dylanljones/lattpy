# coding: utf-8
"""
Created on 15 May 2020
author: Dylan Jones
"""
import numpy as np
import copy


class LatticeCache:

    def __init__(self):
        self.indices = None
        self.neighbours = None
        self.periodic_neighbours = None

    @property
    def n(self):
        return len(self.indices) if self.indices is not None else 0

    def copy(self):
        data = LatticeCache()
        if self.indices is not None:
            indices = self.indices.copy()
            neighbours = copy.deepcopy(self.neighbours)
            data.set(indices, neighbours)
        if self.periodic_neighbours is not None:
            data.set_periodic_neighbours(copy.deepcopy(self.periodic_neighbours))
        return data

    def reset(self):
        self.indices = None
        self.neighbours = None
        self.periodic_neighbours = None

    def set(self, indices, neighbours):
        self.indices = np.asarray(indices)
        self.neighbours = neighbours

    def set_indices(self, indices):
        self.indices = np.asarray(indices)

    def set_neighbours(self, neighbours):
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

    def __bool__(self):
        return self.indices is not None
