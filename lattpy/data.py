# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""This module contains objects for storing and handling the data of finite lattices.

Classes
-------

NeighbourMap:
    Used for saving the mapping of different sites as neighbours.
    Also supports periodic neighbours.

LatticeData:
    Used for saving the site-indices, site-positions and a `NeighbourMap` of
    a (finite) Lattice.
"""

import itertools
import numpy as np
from collections.abc import Mapping
from typing import Iterator, Optional, List, Set, Tuple, Iterable, Union, Sequence


# =========================================================================
# Neighbour mapping
# =========================================================================


class NeighbourMap(Mapping):
    """Object for saving the neighbour mappings (normal and periodic) of lattice sites."""

    __slots__ = ["_neighbours", "_periodic"]

    def __init__(self, size: Optional[int] = 0, num_dist: Optional[int] = 1):
        """Creates a new `NeighbourMap` instance, see the `ìnit`-method for initialization."""
        super().__init__()
        self._neighbours = np.array([[set()]], dtype=set)
        self._periodic = dict()
        if size:
            self.init(size, num_dist)

    @property
    def shape(self) -> Tuple[int, int]:
        """The shape of the neighbour map."""
        return self._neighbours.shape

    @property
    def num_sites(self) -> int:
        """The number of sites in the neighbour map."""
        return self._neighbours.shape[0]

    @property
    def num_dist(self) -> int:
        """The number of distances in the neighbour map."""
        return self._neighbours.shape[1]

    def init(self, size: Optional[int], num_dist: Optional[int] = 1) -> None:
        """Initialize the neighbours map with empty sets.

        If `num_dist==1` only nearest neighbours can be stored, if `num_dist==2'
        nearest and next-nearest neighbours can be stored and so on. The map has
        to be initialized before setting neighbours.
        A `ndarray` of `set` is used since the number of neighbours per site can
        not be estimated in some cases. Set ensures unique indices and supports
        adding more indices.

        Parameters
        ----------
        size: int
            The number of items (sites) of the `NeighbourMap`.
        num_dist: int, optional
            The maximum number of distances of the neighbours.
            The default is `1`(nearest neighbours).
        """
        empty_neighbours = [[set() for _ in range(num_dist)] for _ in range(size)]
        self._neighbours = np.array(empty_neighbours, dtype=set)
        self._periodic = dict()

    def copy(self) -> 'NeighbourMap':
        """Creates a deep copy of the `NeighbourMap` instance."""
        new = self.__class__()
        new._neighbours = self._neighbours.copy()
        new._periodic = self._periodic.copy()
        return new

    def set(self, data: Iterable[Iterable[Iterable[int]]]) -> None:
        """Set the neighbour data."""
        data = np.array(data, dtype=set)
        assert len(data.shape) == 2
        self._neighbours = data

    def add(self, site: int, neighbour: int, distidx: Optional[int] = 0,
            symmetric: Optional[bool] = False) -> None:
        """Add a neighbour index to a site.

        Parameters
        ----------
        site: int
            The index of the site where the neighbour is added.
        neighbour: int
            The index of the site that will be added as neighbour.
        distidx: int, optional
            The index of the distance between the sites.
            The default is `0` (nearest neighbours)
        symmetric: bool, optional
            Optional flag if the neighbour should be set symmetric.
            If `True` the `site` index will be added as neighbour to
            the site with the `neighbour` index. The default is `False`.
        """
        # shift indices to [0, ..., n], so that negative indices are valid.
        site = self._neighbours.shape[0] + site if site < 0 else site
        neighbour = self._neighbours.shape[0] + neighbour if neighbour < 0 else neighbour
        # Add neighbour(s)
        self._neighbours[site, distidx].add(neighbour)
        if symmetric:
            self._neighbours[neighbour, distidx].add(site)

    def add_periodic(self, site: int, neighbour: int, distidx: Optional[int] = 0,
                     symmetric: Optional[bool] = False) -> None:
        """Add a periodic neighbour index to a site.

        Parameters
        ----------
        site: int
            The index of the site where the neighbour is added.
        neighbour: int
            The index of the site that will be added as neighbour.
        distidx: int, optional
            The index of the distance between the sites.
            The default is `0` (nearest neighbours)
        symmetric: bool, optional
            Optional flag if the neighbour should be set symmetric.
            If `True` the `site` index will be added as neighbour to
            the site with the `neighbour` index. The default is `False`.
        """
        # shift indices to [0, ..., n], so that negative indices are valid.
        site = self._neighbours.shape[0] + site if site < 0 else site
        neighbour = self._neighbours.shape[0] + neighbour if neighbour < 0 else neighbour
        # Add neighbour(s)
        self.add(site, neighbour, distidx, symmetric=symmetric)
        # Add periodic neighbour(s)
        num_dist = self._neighbours.shape[1]
        self._periodic.setdefault(site, [set() for _ in range(num_dist)])[distidx].add(neighbour)
        if symmetric:
            self._periodic.setdefault(neighbour, [set() for _ in range(num_dist)])[distidx].add(site)

    def remove(self, site: int, neighbour: int, distidx: Optional[int] = 0,
               symmetric: Optional[bool] = False) -> None:
        """Removes a neighbour from a site.

        Parameters
        ----------
        site: int
            The index of the site where the neighbour is removed.
        neighbour: int
            The index of the site that will be removed as neighbour.
        distidx: int, optional
            The index of the distance between the sites.
            The default is `0` (nearest neighbours)
        symmetric: bool, optional
            Optional flag if the neighbour should be removed symmetric.
            If `True` the `site` index will be removed as neighbour from
            the site with the `neighbour` index. The default is `False`.
        """
        # shift indices to [0, ..., n], so that negative indices are valid.
        site = self._neighbours.shape[0] + site if site < 0 else site
        neighbour = self._neighbours.shape[0] + neighbour if neighbour < 0 else neighbour
        # remove neighbour(s)
        self._neighbours[site, distidx].remove(neighbour)
        if symmetric:
            self._neighbours[neighbour, distidx].remove(site)

    def remove_site_neighbours(self, site, symmetric=True):
        """Removes all neighbours from a site.

        Parameters
        ----------
        site: int
            The index of the site where the neighbours are removed.
        symmetric: bool, optional
            Optional flag if the neighbours should be removed symmetric.
            If `True` the `site` index will be removed as neighbour from
            the site with the `neighbour` index. The default is `True`.
        """
        # shift indices to [0, ..., n], so that negative indices are valid.
        site = self._neighbours.shape[0] + site if site < 0 else site
        for distidx in range(self.num_dist):
            neighbours = self._neighbours[site, distidx].copy()
            for j in neighbours:
                self.remove(site, j, distidx, symmetric)

    def remove_periodic(self, site: int, neighbour: int, distidx: Optional[int] = 0,
                        symmetric: Optional[bool] = False) -> None:
        """Removes a periodic neighbour from a site.

        Parameters
        ----------
        site: int
            The index of the site where the periodic neighbour is removed.
        neighbour: int
            The index of the site that will be removed as periodic neighbour.
        distidx: int, optional
            The index of the distance between the sites.
            The default is `0` (nearest neighbours)
        symmetric: bool, optional
            Optional flag if the neighbour should be removed symmetric.
            If `True` the `site` index will be removed as neighbour from
            the site with the `neighbour` index. The default is `False`.
        """
        # shift indices to [0, ..., n], so that negative indices are valid.
        site = self._neighbours.shape[0] + site if site < 0 else site
        neighbour = self._neighbours.shape[0] + neighbour if neighbour < 0 else neighbour
        # remove neighbour(s)
        self.remove(site, neighbour, distidx, symmetric=symmetric)
        # remove periodic neighbour(s)
        if site in self._periodic:
            self._periodic[site][distidx].remove(neighbour)
        if symmetric and neighbour in self._periodic:
            self._periodic[neighbour][distidx].remove(site)

    def remove_all_periodic(self) -> None:
        """Removes all periodic neighbours."""
        for site, array in self._periodic.items():
            for distidx, neighbours in enumerate(array):
                for neighbour in neighbours:
                    self.remove(site, neighbour, distidx)
        self._periodic = dict()

    def get(self, site: int, distidx: Optional[int] = 0) -> Iterable[int]:
        """Gets the neighbours for a specific distance of a site.

        Parameters
        ----------
        site: int
            The index of the site.
        distidx: int, optional
            The index of the distance. The default is `0` (nearest neighbours).

        Returns
        -------
        neighbours: set of int
            The indices of the neighbours.
        """
        return self._neighbours[site, distidx]

    def getall(self, site: int) -> List[int]:
        """Gets all neighbours of a site.

        Parameters
        ----------
        site: int
            The index of the site.

        Returns
        -------
        neighbours: set of int
            The indices of the neighbours.
        """
        return list(itertools.chain.from_iterable(self._neighbours[site]))

    def get_periodic(self, site: int, distidx: Optional[int] = 0) -> Set[int]:
        """Gets only the periodic neighbours for a specific distance of a site.

        Parameters
        ----------
        site: int
            The index of the site.
        distidx: int, optional
            The index of the distance. The default is `0` (nearest neighbours).

        Returns
        -------
        periodic: set of int
            The indices of the periodic neighbours.
        """
        if site in self._periodic:
            return self._periodic[site][distidx]
        return set()

    def getall_periodic(self, site: int) -> List[int]:
        """Gets all periodic neighbours of a site.

        Parameters
        ----------
        site: int
            The index of the site.

        Returns
        -------
        neighbours: set of int
            The indices of the neighbours.
        """
        if site in self._periodic:
            return list(itertools.chain.from_iterable(self._periodic[site]))
        return list()

    def get_nonperiodic(self, site: int, distidx: Optional[int] = 0) -> Iterable[int]:
        """Gets all neighbours for a specific distance of a site that are not periodic.

        Parameters
        ----------
        site: int
            The index of the site.
        distidx: int, optional
            The index of the distance. The default is `0` (nearest neighbours).

        Returns
        -------
        nonperiodic: set of int
            The indices of the non-periodic neighbours.
        """
        neighbours = set(self.get(site, distidx))
        if site in self._periodic:
            for periodic in self._periodic[site][distidx]:
                if periodic in neighbours:
                    neighbours.remove(periodic)
        return neighbours

    def check_symmetry(self) -> bool:
        """Check if all neighbours of the `NeighbourMap` are symmetric.

        Returns `False` if there exist any neighbour-indices for a site,
        where the neighbours don't also contain the site as neighbour.

        Returns
        -------
        symmetric: bool
        """
        for site in range(self._neighbours.shape[0]):
            # check the neighbours of the site
            for distidx in range(self._neighbours.shape[1]):
                for i in self._neighbours[site, distidx]:
                    # If unsymmetric, return False
                    if site not in self._neighbours[i, distidx]:
                        return False
        return True

    def ensure_neighbour_symmetry(self, site: int) -> None:
        """Ensures the neighbours of a specific site are symmetric.

        Parameters
        ----------
        site: int
            The index of the site.
        """
        # check the neighbours of the site
        for distidx in range(self._neighbours.shape[1]):
            for i in self._neighbours[site, distidx]:
                # Make symmetric if necessary
                if site not in self._neighbours[i, distidx]:
                    self._neighbours[i, distidx].add(site)

    def ensure_symmetry(self) -> None:
        """Ensures the neighbours of all sites are symmetric."""
        for site in range(self._neighbours.shape[0]):
            self.ensure_neighbour_symmetry(site)

    def __getitem__(self, item: Union[float, tuple, list, np.ndarray, slice]) -> Union[float, np.ndarray]:
        """Returns the neighbours of a site."""
        return self._neighbours[item]

    def __len__(self) -> int:
        """Returns the number of sites in the `NeighbourMap`."""
        return len(self._periodic)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterates over the neighbours of all sites."""
        return iter(self._neighbours)

    def __str__(self) -> str:
        """Returns a string representation of the neighbour data."""
        return str(self._neighbours)


# =========================================================================
# Lattice-data
# =========================================================================


class LatticeData:
    """Object for storing the indices, positions and neighbours of lattice sites."""

    __slots__ = ["indices", "positions", "neighbours"]

    def __init__(self):
        self.indices = np.array([])
        self.positions = np.array([])
        self.neighbours = NeighbourMap()

    @property
    def dim(self) -> int:
        """The dimension of the data points."""
        return self.positions.shape[1]

    @property
    def num_sites(self) -> int:
        """The number of sites stored."""
        return len(self.indices)

    @property
    def num_dist(self) -> int:
        """The number of distances of the neighbour data."""
        return self.neighbours.num_dist

    def copy(self) -> 'LatticeData':
        """Creates a deep copy of the instance."""
        data = self.__class__()
        data.indices = self.indices.copy()
        data.positions = self.positions.copy()
        data.neighbours = self.neighbours.copy()
        return data

    def reset(self) -> None:
        """Resets the `LatticeData` instance."""
        self.indices = np.array([])
        self.positions = np.array([])
        self.neighbours = NeighbourMap()

    def set(self, indices: Sequence[Iterable[int]], positions: Sequence[Iterable[float]],
            neighbours: Iterable[Iterable[Iterable[int]]]) -> None:
        """Sets the data of the `LatticeData` instance.

        Parameters
        ----------
        indices: array_like of iterable of int
            The lattice indices of the sites.
        positions: array_like of iterable of int
            The positions of the sites.
        neighbours: iterable of iterable of iterable of int
            The neighbours of the sites.
        """
        self.indices = np.asarray(indices, dtype=np.int16)
        self.positions = np.asarray(positions)
        self.neighbours.set(neighbours)

    def set_positions(self, indices: Sequence[Iterable[int]],
                      positions: Sequence[Iterable[float]]) -> None:
        """Sets the position data of the `LatticeData` instance.

        Parameters
        ----------
        indices: array_like of iterable of int
            The lattice indices of the sites.
        positions: array_like of iterable of int
            The positions of the sites.
        """
        indices = np.arange(len(positions)) if indices is None else indices
        self.indices = np.asarray(indices)
        self.positions = np.asarray(positions)

    def set_neighbours(self, neighbours: Iterable[Iterable[Iterable[int]]]) -> None:
        """Sets the neighbour data of the `LatticeData` instance.

        Parameters
        ----------
        neighbours: iterable of iterable of iterable of int
            The neighbours of the sites.
        """
        self.neighbours.set(neighbours)

    def init_neighbours(self, num_dist: Optional[int] = 1) -> None:
        """Initializes the neighbour map.

        Parameters
        ----------
        num_dist: int, optional
            The maximum number of distances of the neighbours.
            The default is `1`(nearest neighbours).
        """
        self.neighbours.init(self.num_sites, num_dist)

    def add_neighbour(self, site: int, neighbour: int, distidx: Optional[int] = 0,
                      symmetric: Optional[bool] = False) -> None:
        """Adds a neighbour of a site to the neighbour map.

        Raises
        ------
        ValueError:
            A `ValueError`is raised if the neighbour map hasn't been initialized.
            This is needed for allocating the memory of the `np.ndarray`.

        Parameters
        ----------
        site: int
            The index of the site where the neighbour is added.
        neighbour: int
            The index of the site that will be added as neighbour.
        distidx: int, optional
            The index of the distance between the sites.
            The default is `0` (nearest neighbours)
        symmetric: bool, optional
            Optional flag if the neighbour should be set symmetric.
            If `True` the `site` index will be added as neighbour to
            the site with the `neighbour` index. The default is `False`.
        """
        if self.neighbours.num_sites != self.num_sites:
            raise ValueError("Neighbours not initialized!")
        self.neighbours.add(site, neighbour, distidx, symmetric)

    def add_periodic_neighbour(self, site: int, neighbour: int, distidx: Optional[int] = 0,
                               symmetric: Optional[bool] = False) -> None:
        """Adds a periodic neighbour of a site to the neighbour map.

        Raises
        ------
        ValueError:
            A `ValueError`is raised if the neighbour map hasn't been initialized.
            This is needed for allocating the memory of the `np.ndarray`.

        Parameters
        ----------
        site: int
            The index of the site where the neighbour is added.
        neighbour: int
            The index of the site that will be added as neighbour.
        distidx: int, optional
            The index of the distance between the sites.
            The default is `0` (nearest neighbours)
        symmetric: bool, optional
            Optional flag if the neighbour should be set symmetric.
            If `True` the `site` index will be added as neighbour to
            the site with the `neighbour` index. The default is `False`.
        """
        if self.neighbours.num_sites != self.num_sites:
            raise ValueError("Neighbours not initialized!")
        self.neighbours.add_periodic(site, neighbour, distidx, symmetric)

    def get_neighbours(self, site: int, distidx: Optional[int] = 0) -> Iterable[int]:
        """Gets the neighbours for a specific distance of a site.

        Parameters
        ----------
        site: int
            The index of the site.
        distidx: int, optional
            The index of the distance. The default is `0` (nearest neighbours).

        Returns
        -------
        neighbours: set of int
            The indices of the neighbours.
        """
        return self.neighbours.get(site, distidx)

    def get_all_neighbours(self, site: int) -> Iterable[int]:
        """Gets the neighbours for all distances of a site.

        Parameters
        ----------
        site: int
            The index of the site.

        Returns
        -------
        neighbours: set of int
            The indices of the neighbours.
        """
        return self.neighbours.getall(site)

    def get_periodic_neighbours(self, site: int, distidx: Optional[int] = 0) -> Iterable[int]:
        """Gets only the periodic neighbours for a specific distance of a site.

        Parameters
        ----------
        site: int
            The index of the site.
        distidx: int, optional
            The index of the distance. The default is `0` (nearest neighbours).

        Returns
        -------
        periodic: set of int
            The indices of the periodic neighbours.
        """
        return self.neighbours.get_periodic(site, distidx)

    def get_nonperiodic_neighbours(self, site: int, distidx: Optional[int] = 0) -> Iterable[int]:
        """Gets all neighbours for a specific distance of a site that are not periodic.

        Parameters
        ----------
        site: int
            The index of the site.
        distidx: int, optional
            The index of the distance. The default is `0` (nearest neighbours).

        Returns
        -------
        nonperiodic: set of int
            The indices of the non-periodic neighbours.
        """
        return self.neighbours.get_nonperiodic(site, distidx)

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

    def invalidate(self, sites):
        sites = np.atleast_1d(sites).astype(np.int)
        for i in sites:
            self.neighbours.remove_site_neighbours(i, symmetric=True)
            self.indices[i, :] = np.nan
            self.positions[i, :] = np.nan

    def get_invalid(self):
        return np.sort(np.where(np.isnan(self.positions).any(axis=1))[0])

    def find_index(self, indices):
        locs = list()
        indices = np.atleast_2d(indices)
        if len(indices[0]):
            for idx in indices:
                locs.extend(np.where((self.indices == idx).all(axis=1))[0])
        return locs

    def iter_sites(self):
        invalid = self.get_invalid()
        for site in range(len(self.indices)):
            if site not in invalid:
                yield site

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
