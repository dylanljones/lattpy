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
from typing import Union, Optional, Tuple, List, Sequence, Iterable, Set


class NeighbourMap:

    __slots__ = ["data", "periodic"]

    def __init__(self, size: Optional[int] = 0, num_dist: Optional[int] = 1):
        self.data = np.array([[set()]], dtype=set)
        self.periodic = dict()
        if size:
            self.init(size, num_dist)

    @property
    def shape(self) -> Tuple[int]:
        return self.data.shape

    @property
    def num_sites(self) -> int:
        return self.data.shape[0]

    @property
    def num_dist(self) -> int:
        return self.data.shape[1]

    def copy(self) -> 'NeighbourMap':
        new = self.__class__()
        new.data = self.data.copy()
        new.periodic = self.periodic.copy()
        return new

    @staticmethod
    def _empty(num_dist: int) -> List[set]:
        return [set() for _ in range(num_dist)]

    def __str__(self) -> str:
        return str(self.data)

    def __bool__(self) -> bool:
        return self.data.shape[0] > 1

    def __getitem__(self, item: Union[int, tuple, list, np.ndarray, slice]) -> Union[int, float, np.ndarray]:
        return self.data[item]

    def _get_index(self, idx: int) -> int:
        return self.data.shape[0] + idx if idx < 0 else idx

    def set(self, data: Sequence) -> None:
        self.data = np.array(data, dtype=set)

    def init(self, size: int, num_dist: Optional[int] = 1) -> None:
        self.set([self._empty(num_dist) for _ in range(size)])

    def add(self, site: int, neighbour: int, distidx: Optional[int] = 0,
            symmetric: Optional[bool] = False) -> None:
        site = self._get_index(site)
        neighbour = self._get_index(neighbour)
        self.data[site, distidx].add(neighbour)
        if symmetric:
            self.data[neighbour, distidx].add(site)

    def get(self, site: int, distidx: Optional[int] = 0) -> Set[int]:
        return self.data[site, distidx]

    def getall(self, site: int) -> List[int]:
        return list(itertools.chain.from_iterable(self.data[site]))

    def iterall(self, site: int) -> Iterable[int]:
        return itertools.chain.from_iterable(self.data[site])

    def remove(self, site: int, neighbour: int, distidx: Optional[int] = 0) -> None:
        self.data[site, distidx].remove(neighbour)

    # =========================================================================

    def add_peridic(self, site: int, neighbour: int, distidx: Optional[int] = 0,
                    symmetric: Optional[bool] = False) -> None:
        site = self._get_index(site)
        neighbour = self._get_index(neighbour)

        self.add(site, neighbour, distidx, symmetric=symmetric)
        num_dist = self.data.shape[1]
        self.periodic.setdefault(site, self._empty(num_dist))[distidx].add(neighbour)
        if symmetric:
            self.periodic.setdefault(neighbour, self._empty(num_dist))[distidx].add(site)

    def get_periodic(self, site: int, distidx: Optional[int] = 0) -> Set[int]:
        if site in self.periodic:
            return self.periodic[site][distidx]
        return set()

    def get_non_periodic(self, site: int, distidx: Optional[int] = 0) -> Set[int]:
        neighbours = self.get(site, distidx)
        if site in self.periodic:
            for periodic in self.periodic[site][distidx]:
                if periodic in neighbours:
                    neighbours.remove(periodic)
        return neighbours

    def get_all_periodic(self, site: int) -> List[int]:
        if site in self.periodic:
            return list(itertools.chain.from_iterable(self.periodic[site]))
        return list()

    def remove_periodic(self, site: int, neighbour: int, distidx: Optional[int] = 0) -> None:
        if site in self.periodic:
            self.periodic[site][distidx].remove(neighbour)
        self.remove(site, neighbour, distidx)

    def remove_all_periodic(self) -> None:
        for site, array in self.periodic.items():
            for distidx, neighbours in enumerate(array):
                for neighbour in neighbours:
                    self.remove(site, neighbour, distidx)
        self.periodic = dict()

    # =========================================================================

    def _find_unsymmetric(self, site: int) -> List[Tuple[int, int]]:
        missing = list()
        for distidx in range(self.data.shape[1]):
            for i in self.data[site, distidx]:
                if site not in self.data[i, distidx]:
                    missing.append((i, distidx))
        return missing

    def check_symmetry(self) -> bool:
        for site in range(self.data.shape[0]):
            if self._find_unsymmetric(site):
                return False
        return True

    def ensure_neighbour_symmetry(self, site: int) -> None:
        for i, distidx in self._find_unsymmetric(site):
            self.data[i, distidx].add(site)

    def ensure_symmetry(self) -> None:
        for site in range(self.data.shape[0]):
            self.ensure_neighbour_symmetry(site)


class LatticeData:

    __slots__ = ["indices", "positions", "neighbours"]

    def __init__(self, *args):  # noqa
        self.indices = np.array([])
        self.positions = np.array([])
        self.neighbours = NeighbourMap()

    @property
    def dim(self) -> int:
        return self.positions.shape[1]

    @property
    def num_sites(self) -> int:
        return len(self.indices)

    @property
    def num_dist(self) -> int:
        return self.neighbours.num_dist

    def copy(self) -> 'LatticeData':
        data = self.__class__()
        data.indices = self.indices.copy()
        data.positions = self.positions.copy()
        data.neighbours = self.neighbours.copy()
        return data

    def __bool__(self) -> bool:
        return bool(len(self.indices))

    def __str__(self) -> str:
        widths = 9, 15, 10
        delim = " | "
        headers = "Indices", "Positions", "Neighbours"
        lines = list()
        lines.append(f"{headers[0]:<{widths[0]}}{delim}{headers[1]:<{widths[1]}}{delim}{headers[2]}")
        for site in range(self.num_sites):
            pos = "[" + ", ".join(f"{x:.1f}" for x in self.positions[site]) + "]"
            idx = str(self.indices[site])
            neighbours = str(self.neighbours[site])
            lines.append(f"{idx:<{widths[0]}}{delim}{pos:<{widths[1]}}{delim}{neighbours}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.indices = np.array([])
        self.positions = np.array([])
        self.neighbours = NeighbourMap()

    def set(self, indices, positions, neighbours) -> None:
        self.indices = np.asarray(indices, dtype=np.int16)
        self.positions = np.asarray(positions)
        self.neighbours.set(neighbours)

    def set_positions(self, positions, indices=None) -> None:
        indices = np.arange(len(positions)) if indices is None else indices
        self.indices = np.asarray(indices)
        self.positions = np.asarray(positions)

    def set_neighbours(self, neighbours) -> None:
        self.neighbours.set(neighbours)

    def init_neighbours(self, num_dist=1) -> None:
        self.neighbours.init(self.num_sites, num_dist)

    def add_neighbour(self, site, neighbours, distidx=0, symmetric=False) -> None:
        if self.neighbours.num_sites != self.num_sites:
            raise ValueError("Neighbours not initialized!")
        self.neighbours.add(site, neighbours, distidx, symmetric)

    def add_periodic_neighbour(self, site, neighbours, distidx=0, symmetric=False) -> None:
        if self.neighbours.num_sites != self.num_sites:
            raise ValueError("Neighbours not initialized!")
        self.neighbours.add_peridic(site, neighbours, distidx, symmetric)

    def get_neighbours(self, site, distidx=0) -> Set[int]:
        return self.neighbours.get(site, distidx)

    def get_all_neighbours(self, site) -> List[int]:
        return self.neighbours.getall(site)

    def get_periodic_neighbours(self, site, distidx=0) -> Set[int]:
        return self.neighbours.get_periodic(site, distidx)

    def get_nonperiodic_neighbours(self, site, distidx=0) -> Set[int]:
        return self.neighbours.get_non_periodic(site, distidx)

    def get_limits(self) -> np.ndarray:
        return np.array([np.min(self.positions, axis=0), np.max(self.positions, axis=0)])

    def get_index_limits(self) -> np.ndarray:
        return np.array([np.min(self.indices, axis=0), np.max(self.indices, axis=0)])

    def site_mask(self, mins=(), maxs=(), invert: Optional[bool] = False) -> np.ndarray:
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
        return np.asarray(1 - mask if invert else mask)

    def find_sites(self, mins=(), maxs=(), invert: Optional[bool] = False) -> np.ndarray:
        mask = self.site_mask(mins, maxs, invert)
        return np.where(mask)[0]

    def find_outer_sites(self, ax: int, offset: int) -> np.ndarray:
        limits = self.get_limits()
        mins = [None for _ in range(self.dim)]
        maxs = [None for _ in range(self.dim)]
        mins[ax] = limits[0, ax] + offset
        maxs[ax] = limits[1, ax] - offset
        mask = self.site_mask(mins, maxs, invert=True)
        return np.where(mask)[0]
