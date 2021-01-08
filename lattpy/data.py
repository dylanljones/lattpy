# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""This module contains objects for low-level representation of lattice systems."""

import numpy as np
from copy import deepcopy
from typing import Optional, Iterable, Union, Sequence
from .utils import create_lookup_table
import logging

__all__ = ["DataMap", "LatticeData"]

logging.captureWarnings(True)

logger = logging.getLogger(__name__)


class DataMap:

    def __init__(self, alphas: np.ndarray, pairs: np.ndarray, distindices: np.ndarray):
        sites = np.arange(len(alphas), dtype=pairs.dtype)
        self._map = np.append(-alphas-1, distindices)
        self._indices = np.append(np.tile(sites, (2, 1)).T, pairs, axis=0)

    @property
    def size(self):
        return len(self._indices)

    @property
    def indices(self):
        return self._indices.T

    @property
    def rows(self):
        return self._indices[:, 0]

    @property
    def cols(self):
        return self._indices[:, 1]

    @property
    def nbytes(self):
        """Returns the number of bytes stored."""
        return self._map.nbytes + self._indices.nbytes

    def onsite(self, alpha):
        return self._map == -alpha-1

    def hopping(self, distidx):
        return self._map == distidx

    def fill(self, a, hop, eps=0., copy=False):
        out = a.copy() if copy else a
        eps = np.atleast_1d(eps)
        hop = np.atleast_1d(hop)
        for alpha, value in enumerate(eps):
            out[self.onsite(alpha)] = value
        for dist, value in enumerate(hop):
            out[self.hopping(dist)] = value
        return out


class LatticeData:
    """Object for storing the indices, positions and neighbours of lattice sites."""

    def __init__(self, *args):
        self.indices = np.array([])
        self.positions = np.array([])
        self.neighbours = np.array([])
        self.distances = np.array([])
        self.distvals = np.array([])
        self.paxes = np.array([])

        self.invalid_idx = -1
        self.invalid_distidx = -1

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
        """The number of distances of the neighbour data."""
        return len(np.unique(self.distances[np.isfinite(self.distances)]))

    @property
    def nbytes(self):
        """Returns the number of bytes stored."""
        size = self.indices.nbytes + self.positions.nbytes
        size += self.neighbours.nbytes + self.distances.nbytes
        size += self.distvals.nbytes + self.paxes.nbytes
        return size

    def copy(self) -> 'LatticeData':
        """Creates a deep copy of the instance."""
        return deepcopy(self)

    def reset(self) -> None:
        """Resets the `LatticeData` instance."""
        self.indices = np.array([])
        self.positions = np.array([])
        self.neighbours = np.array([])
        self.distances = np.array([])
        self.distvals = np.array([])
        self.paxes = np.array([])
        self.invalid_idx = -1
        self.invalid_distidx = -1

    def set(self, indices: Sequence[Iterable[int]],
            positions: Sequence[Iterable[float]],
            neighbours: Iterable[Iterable[Iterable[int]]],
            distances: Iterable[Iterable[Iterable[float]]]) -> None:
        """Sets the data of the `LatticeData` instance.

        Parameters
        ----------
        indices: array_like of iterable of int
            The lattice indices of the sites.
        positions: array_like of iterable of int
            The positions of the sites.
        neighbours: iterable of iterable of of int
            The neighbours of the sites.
        distances: iterabe of iterable of int
            The distances of the neighbours.
        """
        logger.debug("Setting data")
        distvals, distidx = create_lookup_table(distances)

        self.indices = indices
        self.positions = positions
        self.neighbours = neighbours
        self.distances = distidx
        self.distvals = distvals
        self.paxes = np.ones_like(self.distances) * self.dim

        self.invalid_idx = self.num_sites
        self.invalid_distidx = np.max(self.distances)

    def get_limits(self) -> np.ndarray:
        """Computes the geometric limits of the positions of the stored sites.

        Returns
        -------
        limits: np.ndarray
            The minimum and maximum value for each axis of the position data.
        """
        return np.array([np.min(self.positions, axis=0), np.max(self.positions, axis=0)])

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
        limits: np.ndarray
            The minimum and maximum value for each axis of the lattice indices.
        """
        return self.get_index_limits()[:, :-1]

    def neighbour_mask(self, site: int, distidx: Optional[int] = None,
                       periodic: Optional[bool] = None,
                       unique: Optional[bool] = False) -> np.ndarray:
        """Creates a mask for the valid neighbours of a specific site.

        Parameters
        ----------
        site: int
            The index of the site.
        distidx: int, optional
            The index of the distance. If ``None`` the data for all distances is returned.
            The default is `None` (all neighbours).
        periodic: bool, optional
            Periodic neighbour flag. If ``None`` the data for all neighbours is returned.
            If a bool is passed either the periodic or non-periodic neighbours are masked.
            The default is ``None`` (all neighbours).
        unique: bool, optional
            If 'True', each unique pair is only return once. The defualt is ``False``.

        Returns
        -------
        mask: np.ndarray
        """
        if distidx is None:
            mask = self.distances[site] < self.invalid_distidx
        else:
            mask = self.distances[site] == distidx
        if unique:
            mask &= self.neighbours[site] > site

        if periodic is not None:
            if periodic:
                mask &= self.paxes[site] != self.dim
            else:
                mask &= self.paxes[site] == self.dim
        return mask

    def set_periodic(self, indices: dict, distances: dict, axes: dict) -> None:
        """ Adds periodic neighbours to the invalid slots of the neighbour data

        Parameters
        ----------
        indices: dict
            Indices of the periodic neighbours.
        distances: dict
            The distances of the periodic neighbours.
        axes: dict
            Index of the translation axis of the periodic neighbours.
        """
        for i, pidx in indices.items():
            # compute invalid slots of normal data
            # and remove previous periodic neighbours
            i0 = len(self.get_neighbours(i, periodic=False))
            i1 = i0 + len(pidx)
            self.paxes[i, i0:] = self.dim
            # translate distances to indices
            dists = distances[i]
            distidx = [np.searchsorted(self.distvals, d) for d in dists]
            # add periodic data
            self.neighbours[i, i0:i1] = pidx
            self.distances[i, i0:i1] = distidx
            self.paxes[i, i0:i1] = axes[i]

    def get_positions(self, alpha):
        """Returns the atom positions of a sublattice."""
        mask = self.indices[:, -1] == alpha
        return self.positions[mask]

    def get_neighbours(self, site: int, distidx: Optional[int] = None,
                       periodic: Optional[bool] = None,
                       unique: Optional[bool] = False) -> np.ndarray:
        """Returns all neighbours or the neighbours for a certain distance of a lattice site.

        See the `neighbour_mask`-method for more information on parameters

        Returns
        -------
        neighbours: np.ndarray
            The indices of the neighbours.
        """
        mask = self.neighbour_mask(site, distidx, periodic, unique)
        return self.neighbours[site, mask]

    def get_neighbour_pos(self, site: int, distidx: Optional[int] = None,
                          periodic: Optional[bool] = None,
                          unique: Optional[bool] = False) -> np.ndarray:
        """Returns the neighbour positions of a lattice site.

        See the `neighbour_mask`-method for more information on parameters

        Returns
        -------
        neighbour_positions: np.ndarray
            The positions of the neighbours.
        """
        return self.positions[self.get_neighbours(site, distidx, periodic, unique)]

    def iter_neighbours(self, site: int, unique: Optional[bool] = False) -> np.ndarray:
        """Iterates over the neighbours of all distance levels.

        See the `neighbour_mask`-method for more information on parameters

        Yields
        -------
        distidx: int
        neighbours: np.ndarray
        """
        for distidx in np.unique(self.distances[site]):
            if distidx != self.invalid_distidx:
                yield distidx, self.get_neighbours(site, distidx, unique=unique)

    def map(self) -> DataMap:
        """ Builds a map containing the atom-indices, site-pairs and corresponding distances.

        Returns
        -------
        datamap: DataMap
        """
        alphas = self.indices[:, -1].astype(np.int8)

        # Build index pairs and corresponding distance array
        dtype = np.min_scalar_type(self.num_sites)
        sites = np.arange(self.num_sites, dtype=dtype)
        sites_t = np.tile(sites, (self.neighbours.shape[1], 1)).T
        pairs = np.reshape([sites_t, self.neighbours], newshape=(2, -1)).T
        distindices = self.distances.flatten()

        # Filter pairs with invalid indices
        mask = distindices != self.invalid_distidx
        pairs = pairs[mask]
        distindices = distindices[mask]
        return DataMap(alphas, pairs.astype(dtype), distindices)

    def get_all_neighbours(self, site: int) -> Iterable[int]:
        """Gets the neighbours for all distances of a site."""
        logging.warning("DeprecationWarning: "
                        "Use `get_neighbours(site, distidx=None)` instead!")
        return self.get_neighbours(site)

    def get_periodic_neighbours(self, site: int, distidx: Optional[int] = 0) -> Iterable[int]:
        """Gets only the periodic neighbours for a specific distance of a site."""
        logging.warning("DeprecationWarning: "
                        "Use `get_neighbours(site, distidx, periodic=True)` instead!")
        return self.get_neighbours(site, distidx, periodic=True)

    def get_nonperiodic_neighbours(self, site: int, distidx: Optional[int] = 0) -> Iterable[int]:
        """Gets all neighbours for a specific distance of a site that are not periodic."""
        logging.warning("DeprecationWarning: "
                        "Use `get_neighbours(site, distidx, periodic=False)` instead!")
        return self.get_neighbours(site, distidx, periodic=True)

    def site_mask(self, mins: Optional[Sequence[Union[float, None]]] = None,
                  maxs: Optional[Sequence[Union[float, None]]] = None,
                  invert: Optional[bool] = False) -> np.ndarray:
        """Creates a mask for the position data of the sites.

        Parameters
        ----------
        mins: sequence or float or None, optional
            Optional lower bound for the positions. The default is no lower bound.
        maxs: sequence or float or None, optional
            Optional upper bound for the positions. The default is no upper bound.
        invert: bool, optional
            If `True`, the mask is inverted. The default is `False`.

        Returns
        -------
        mask: np.ndarray
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
        mins: sequence or float or None, optional
            Optional lower bound for the positions. The default is no lower bound.
        maxs: sequence or float or None, optional
            Optional upper bound for the positions. The default is no upper bound.
        invert: bool, optional
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
        ax: int
            The geometrical axis.
        offset: int
            The width of the outer slices.

        Returns
        -------
        indices: np.ndarray
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
        widths = 9, 15, 10
        delim = " | "
        headers = "Indices", "Positions", "Neighbours"
        lines = list()
        s = f"{headers[0]:<{widths[0]}}{delim}{headers[1]:<{widths[1]}}{delim}{headers[2]}"
        lines.append(s)
        for site in range(self.num_sites):
            pos = "[" + ", ".join(f"{x:.1f}" for x in self.positions[site]) + "]"
            idx = str(self.indices[site])
            neighbours = str(self.neighbours[site])
            lines.append(f"{idx:<{widths[0]}}{delim}{pos:<{widths[1]}}{delim}{neighbours}")
        return "\n".join(lines)
