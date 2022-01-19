# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""This module contains objects for low-level representation of lattice systems."""

import logging
from copy import deepcopy
from typing import Optional, Iterable, Union, Sequence
import numpy as np
from .utils import ArrayLike, create_lookup_table

__all__ = ["DataMap", "LatticeData"]

logging.captureWarnings(True)

logger = logging.getLogger(__name__)


class DataMap:
    """Object for low-level representation of sites and site-pairs.

    Parameters
    ---------
    alphas : (N) np.ndarray
        The atom indices of the sites.
    pairs : (M, 2) np.ndarray
        An array of index-pairs of the lattice sites.
    distindices : (M) np.ndarray
        The distance-indices for each pair
    """

    def __init__(self, alphas: np.ndarray, pairs: np.ndarray, distindices: np.ndarray):
        sites = np.arange(len(alphas), dtype=pairs.dtype)
        self._map = np.append(-alphas - 1, distindices)
        self._indices = np.append(np.tile(sites, (2, 1)).T, pairs, axis=0)

    @property
    def size(self) -> int:
        """The number of the data points (sites + neighbor pairs)"""
        return len(self._indices)

    @property
    def indices(self) -> np.ndarray:
        """The indices of the data points as rows and collumns."""
        return self._indices.T

    @property
    def rows(self):
        """The rows of the data points."""
        return self._indices[:, 0]

    @property
    def cols(self):
        """The columns of the data points."""
        return self._indices[:, 1]

    @property
    def nbytes(self):
        """The number of bytes stored in the datamap."""
        return self._map.nbytes + self._indices.nbytes

    def onsite(self, alpha: Optional[int] = None) -> np.ndarray:
        """Creates a mask of the site elements for the atoms with the given index.

        Parameters
        ----------
        alpha : int, optional
            Index of the atom in the unitcell. If `None` a mask for all atoms
            is returned. The default is `None`.
        Returns
        -------
        mask : np.ndarray
        """
        if alpha is None:
            return self._map < 0
        return self._map == -alpha - 1

    def hopping(self, distidx: Optional[int] = None) -> np.ndarray:
        """Creates a mask of the site-pair elements with the given distance index.

        Parameters
        ----------
        distidx : int, optional
            Index of distance to neighboring sites, default is 0 (nearest neighbors).
            If `None` a mask for neighbor-connections is returned. The default is
            `None`.
        Returns
        -------
        mask : np.ndarray
        """
        if distidx is None:
            return self._map >= 0
        return self._map == distidx

    def fill(self, array: np.ndarray, hop: ArrayLike,
             eps: Optional[ArrayLike] = 0.) -> np.ndarray:
        """Fills a data-array with the given values mapped to the right indices.

        Parameters
        ----------
        array : np.ndarray
            The array to add the values. The length of the array must match the
            size of the `DataMap`-instance.
        hop : array_like
            The values are used for the site-pairs. The first value corresponds to
            nearest neighbor hopping, the second to next-nearest neighbors and so on.
        eps : array_like, optional
            The onsite values used for the lattice sites. If there are multiple atoms
            in the unitcell the length of the values must match. The default is 0.
        Returns
        -------
        filled : np.ndarray
        """
        eps = np.atleast_1d(eps)
        hop = np.atleast_1d(hop)
        for alpha, value in enumerate(eps):
            array[self.onsite(alpha)] = value
        for dist, value in enumerate(hop):
            array[self.hopping(dist)] = value
        return array


class LatticeData:
    """Object for storing the indices, positions and neighbors of lattice sites.

    Parameters
    ----------
    indices : array_like of iterable of int
        The lattice indices of the sites.
    positions : array_like of iterable of int
        The positions of the sites.
    neighbors : iterable of iterable of of int
        The neighbors of the sites.
    distances : iterabe of iterable of int
        The distances of the neighbors.
    """

    def __init__(self, *args):
        self.indices = np.array([])
        self.positions = np.array([])
        self.neighbors = np.array([])
        self.distances = np.array([])

        self.distvals = np.array([])
        self.paxes = np.array([])

        self.invalid_idx = -1
        self.invalid_distidx = -1
        self._dmap = None

        if args:
            self.set(*args)

    @property
    def dim(self) -> int:
        """The dimension of the data points."""
        return self.positions.shape[1]

    @property
    def num_sites(self) -> int:
        """The number of sites stored."""
        return self.indices.shape[0]

    @property
    def num_distances(self) -> int:
        """The number of distances of the neighbor data."""
        return len(np.unique(self.distances[np.isfinite(self.distances)]))

    @property
    def nbytes(self):
        """Returns the number of bytes stored."""
        size = self.indices.nbytes + self.positions.nbytes
        size += self.neighbors.nbytes + self.distances.nbytes
        size += self.distvals.nbytes + self.paxes.nbytes
        return size

    def copy(self) -> 'LatticeData':
        """Creates a deep copy of the instance."""
        return deepcopy(self)

    def reset(self) -> None:
        """Resets the `LatticeData` instance."""
        self.indices = np.array([])
        self.positions = np.array([])
        self.neighbors = np.array([])
        self.distances = np.array([])
        self.distvals = np.array([])
        self.paxes = np.array([])
        self._dmap = None
        self.invalid_idx = -1
        self.invalid_distidx = -1

    def set(self, indices: Sequence[Iterable[int]],
            positions: Sequence[Iterable[float]],
            neighbors: Iterable[Iterable[Iterable[int]]],
            distances: Iterable[Iterable[Iterable[float]]]) -> None:
        """Sets the data of the `LatticeData` instance.

        Parameters
        ----------
        indices : array_like of iterable of int
            The lattice indices of the sites.
        positions : array_like of iterable of int
            The positions of the sites.
        neighbors : iterable of iterable of of int
            The neighbors of the sites.
        distances : iterabe of iterable of int
            The distances of the neighbors.
        """
        logger.debug("Setting data")
        distvals, distidx = create_lookup_table(distances)

        self.indices = indices
        self.positions = positions
        self.neighbors = neighbors
        self.distances = distidx
        self.distvals = distvals
        self.paxes = np.full_like(self.distances, fill_value=self.dim)

        self.invalid_idx = self.num_sites
        self.invalid_distidx = np.max(self.distances)
        self._dmap = None

    def get_limits(self) -> np.ndarray:
        """Computes the geometric limits of the positions of the stored sites.

        Returns
        -------
        limits : np.ndarray
            The minimum and maximum value for each axis of the position data.
        """
        return np.array([np.min(self.positions, axis=0),
                         np.max(self.positions, axis=0)])

    def get_index_limits(self) -> np.ndarray:
        """Computes the geometric limits of the lattice indices of the stored sites.

        Returns
        -------
        limits: np.ndarray
            The minimum and maximum value for each axis of the lattice indices.
        """
        return np.array([np.min(self.indices, axis=0), np.max(self.indices, axis=0)])

    def get_translation_limits(self) -> np.ndarray:
        """Computes the geometric limits of the translation vectors of the stored sites.

        Returns
        -------
        limits : np.ndarray
            The minimum and maximum value for each axis of the lattice indices.
        """
        return self.get_index_limits()[:, :-1]

    def neighbor_mask(self, site: int, distidx: Optional[int] = None,
                      periodic: Optional[bool] = None,
                      unique: Optional[bool] = False) -> np.ndarray:
        """Creates a mask for the valid neighbors of a specific site.

        Parameters
        ----------
        site : int
            The index of the site.
        distidx : int, optional
            The index of the distance. If `None` the data for all distances is returned.
            The default is `None` (all neighbors).
        periodic : bool, optional
            Periodic neighbor flag. If `None` the data for all neighbors is returned.
            If a bool is passed either the periodic or non-periodic neighbors
            are masked. The default is `None` (all neighbors).
        unique : bool, optional
            If 'True', each unique pair is only return once. The defualt is `False`.

        Returns
        -------
        mask : np.ndarray
        """
        if distidx is None:
            mask = self.distances[site] < self.invalid_distidx
        else:
            mask = self.distances[site] == distidx
        if unique:
            mask &= self.neighbors[site] > site

        if periodic is not None:
            if periodic:
                mask &= self.paxes[site] != self.dim
            else:
                mask &= self.paxes[site] == self.dim
        return mask

    def set_periodic(self, indices: dict, distances: dict, axes: dict) -> None:
        """Adds periodic neighbors to the invalid slots of the neighbor data

        Parameters
        ----------
        indices : dict
            Indices of the periodic neighbors.
        distances : dict
            The distances of the periodic neighbors.
        axes : dict
            Index of the translation axis of the periodic neighbors.
        """
        for i, pidx in indices.items():
            # compute invalid slots of normal data
            # and remove previous periodic neighbors
            i0 = len(self.get_neighbors(i, periodic=False))
            i1 = i0 + len(pidx)
            self.paxes[i, i0:] = self.dim
            # translate distances to indices
            dists = distances[i]
            distidx = [np.searchsorted(self.distvals, d) for d in dists]
            # add periodic data
            self.neighbors[i, i0:i1] = pidx
            self.distances[i, i0:i1] = distidx
            self.paxes[i, i0:i1] = axes[i]

    def sort(self, ax=None, indices=None, reverse=False):
        if ax is not None:
            indices = np.lexsort(self.indices.T[[ax]])
        if reverse:
            indices = indices[::-1]
        # Reorder data
        self.indices = self.indices[indices]
        self.positions = self.positions[indices]
        self.neighbors = self.neighbors[indices]
        self.distances = self.distances[indices]
        self.paxes = self.paxes[indices]
        # Translate neighbor indices
        old_neighbors = self.neighbors.copy()
        for new, old in enumerate(indices):
            mask = old_neighbors == old
            self.neighbors[mask] = new

    def remove_periodic(self):
        mask = self.paxes != self.dim
        self.neighbors[mask] = self.invalid_idx
        self.distances[mask] = self.invalid_distidx
        self.paxes.fill(self.dim)

    def sort_neighbors(self):
        distances = self.distvals[self.distances]
        i = np.arange(len(distances))[:, np.newaxis]
        j = np.argsort(distances, axis=1)
        self.neighbors = self.neighbors[i, j]
        self.distances = self.distances[i, j]
        self.paxes = self.paxes[i, j]

    def add_neighbors(self, site, neighbors, distances):
        neighbors = np.atleast_2d(neighbors)
        distances = np.atleast_2d(distances)
        # compute invalid slots of normal data
        i0 = len(self.distances[site, self.distances[site] != self.invalid_distidx])
        i1 = i0 + len(neighbors)
        # Translate distances to indices
        distidx = [np.searchsorted(self.distvals, d) for d in distances]
        # Add new neighbor data to unused slots
        self.neighbors[site, i0:i1] = neighbors
        self.distances[site, i0:i1] = distidx

    def append(self, *args, copy=False):
        neighbors1 = self.neighbors.copy()
        distances1 = self.distvals[self.distances]
        if len(args) == 1 and isinstance(args[0], LatticeData):
            data = args[0]
            indices2 = data.indices
            positions2 = data.positions
            neighbors2 = data.neighbors
            distances2 = data.distvals[data.distances]
        else:
            indices2, positions2, neighbors2, distances2 = args

        # Remove periodic neighbors
        mask = self.paxes != self.dim
        neighbors1[mask] = self.invalid_idx
        distances1[mask] = self.invalid_distidx
        self.paxes[:] = self.dim

        # Convert invalid indices of neighbor data
        invalid_idx = self.num_sites + len(indices2)
        neighbors1[neighbors1 == self.invalid_idx] = invalid_idx
        neighbors2[neighbors2 == len(indices2)] = invalid_idx

        # Shift neighbor indices
        neighbors2[neighbors2 != invalid_idx] += self.num_sites

        # Pad neighbor data
        cols1 = neighbors1.shape[1]
        cols2 = neighbors2.shape[1]
        cols = max(cols1, cols2)
        if cols1 < cols:
            widths = ((0, 0), (0, cols - cols1))
            neighbors1 = np.pad(neighbors1, widths, constant_values=invalid_idx)
            distances1 = np.pad(distances1, widths, constant_values=np.inf)
        if cols2 < cols:
            widths = ((0, 0), (0, cols - cols2))
            neighbors2 = np.pad(neighbors2, widths, constant_values=invalid_idx)
            distances2 = np.pad(distances2, widths, constant_values=np.inf)

        # Join data
        indices = np.append(self.indices, indices2, axis=0)
        positions = np.append(self.positions, positions2, axis=0)
        neighbors = np.append(neighbors1, neighbors2, axis=0)
        distances = np.append(distances1, distances2, axis=0)

        if copy:
            data = LatticeData(indices, positions, neighbors, distances)
            data.sort_neighbors()
            return data

        self.set(indices, positions, neighbors, distances)
        self.sort_neighbors()
        return self

    def get_positions(self, alpha):
        """Returns the atom positions of a sublattice."""
        mask = self.indices[:, -1] == alpha
        return self.positions[mask]

    def get_neighbors(self, site: int, distidx: Optional[int] = None,
                      periodic: Optional[bool] = None,
                      unique: Optional[bool] = False) -> np.ndarray:
        """Returns the neighbors of a lattice site.

        See the `neighbor_mask`-method for more information on parameters

        Returns
        -------
        neighbors : np.ndarray
            The indices of the neighbors.
        """
        mask = self.neighbor_mask(site, distidx, periodic, unique)
        return self.neighbors[site, mask]

    def get_neighbor_pos(self, site: int, distidx: Optional[int] = None,
                         periodic: Optional[bool] = None,
                         unique: Optional[bool] = False) -> np.ndarray:
        """Returns the neighbor positions of a lattice site.

        See the `neighbor_mask`-method for more information on parameters

        Returns
        -------
        neighbor_positions : np.ndarray
            The positions of the neighbors.
        """
        ind = self.get_neighbors(site, distidx, periodic, unique)
        if np.all(ind == self.invalid_idx):
            return np.array([])
        return self.positions[ind]

    def iter_neighbors(self, site: int, unique: Optional[bool] = False) -> np.ndarray:
        """Iterates over the neighbors of all distance levels.

        See the `neighbor_mask`-method for more information on parameters

        Yields
        -------
        distidx : int
        neighbors : np.ndarray
        """
        for distidx in np.unique(self.distances[site]):
            if distidx != self.invalid_distidx:
                yield distidx, self.get_neighbors(site, distidx, unique=unique)

    def map(self) -> DataMap:
        """Builds a map containing the atom-indices, site-pairs and distances.

        Returns
        -------
        datamap : DataMap
        """
        if self._dmap is None:
            alphas = self.indices[:, -1].astype(np.int8)
            # Build index pairs and corresponding distance array
            dtype = np.min_scalar_type(self.num_sites)
            sites = np.arange(self.num_sites, dtype=dtype)
            sites_t = np.tile(sites, (self.neighbors.shape[1], 1)).T
            pairs = np.reshape([sites_t, self.neighbors], newshape=(2, -1)).T
            distindices = self.distances.flatten()
            # Filter pairs with invalid indices
            mask = distindices != self.invalid_distidx
            pairs = pairs[mask]
            distindices = distindices[mask]
            self._dmap = DataMap(alphas, pairs.astype(dtype), distindices)
        return self._dmap

    def site_mask(self, mins: Optional[Sequence[Union[float, None]]] = None,
                  maxs: Optional[Sequence[Union[float, None]]] = None,
                  invert: Optional[bool] = False) -> np.ndarray:
        """Creates a mask for the position data of the sites.

        Parameters
        ----------
        mins : sequence or float or None, optional
            Optional lower bound for the positions. The default is no lower bound.
        maxs : sequence or float or None, optional
            Optional upper bound for the positions. The default is no upper bound.
        invert : bool, optional
            If `True`, the mask is inverted. The default is `False`.

        Returns
        -------
        mask : np.ndarray
            The mask containing a boolean value for each site.
        """
        if mins is None:
            mins = [None] * self.dim
        elif len(mins) != self.dim:
            mins = list(mins) + list([None] * (self.dim - len(mins)))
        if maxs is None:
            maxs = list([None] * self.dim)
        elif len(maxs) != self.dim:
            maxs = list(maxs) + [None] * (self.dim - len(maxs))
        mins = [(x if x is not None else -np.inf) for x in mins]
        maxs = [(x if x is not None else +np.inf) for x in maxs]
        limits = np.array([mins, maxs])
        mask = 1
        for ax in range(self.dim):
            ax_data = self.positions[:, ax]
            mask &= (limits[0, ax] <= ax_data) & (ax_data <= limits[1, ax])
        return np.asarray(1 - mask if invert else mask)

    def find_sites(self, mins: Optional[Sequence[Union[float, None]]] = None,
                   maxs: Optional[Sequence[Union[float, None]]] = None,
                   invert: Optional[bool] = False) -> np.ndarray:
        """Returns the indices of sites inside or outside the given limits.

        Parameters
        ----------
        mins : sequence or float or None, optional
            Optional lower bound for the positions. The default is no lower bound.
        maxs : sequence or float or None, optional
            Optional upper bound for the positions. The default is no upper bound.
        invert : bool, optional
            If `True`, the mask is inverted and the positions outside of the bounds
            will be returned. The default is `False`.

        Returns
        -------
        indices: np.ndarray
            The indices of the masked sites.
        """
        mask = self.site_mask(mins, maxs, invert)
        return np.where(mask)[0]

    def find_outer_sites(self, ax: int, offset: int) -> np.ndarray:
        """Returns the indices of the outer sites along a specific axis.

        Parameters
        ----------
        ax : int
            The geometrical axis.
        offset : int
            The width of the outer slices.

        Returns
        -------
        indices : np.ndarray
            The indices of the masked sites.
        """
        limits = self.get_limits()
        mins = [None for _ in range(self.dim)]
        maxs = [None for _ in range(self.dim)]
        mins[ax] = limits[0, ax] + offset
        maxs[ax] = limits[1, ax] - offset
        mask = self.site_mask(mins, maxs, invert=True)
        return np.where(mask)[0]

    def __bool__(self) -> bool:
        return bool(len(self.indices))

    def __str__(self) -> str:
        w = 9, 15, 10
        delim = " | "
        headers = "Indices", "Positions", "Neighbours"
        lines = list()
        s = f"{headers[0]:<{w[0]}}{delim}{headers[1]:<{w[1]}}{delim}{headers[2]}"
        lines.append(s)
        for site in range(self.num_sites):
            pos = "[" + ", ".join(f"{x:.1f}" for x in self.positions[site]) + "]"
            idx = str(self.indices[site])
            neighbors = str(self.neighbors[site])
            s = f"{idx:<{w[0]}}{delim}{pos:<{w[1]}}{delim}{neighbors}"
            lines.append(s)
        return "\n".join(lines)
