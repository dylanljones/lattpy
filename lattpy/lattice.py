# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""This module contains the main `Lattice` object."""

import pickle
import logging
import warnings
import itertools
import numpy as np
from copy import deepcopy
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from typing import Union, Optional, Tuple, Iterator, Sequence, Callable
from .utils import ArrayLike, frmt_num, NotBuiltError
from .plotting import (
    subplot,
    draw_sites,
    draw_vectors,
    draw_indices,
    connection_color_array,
)
from .spatial import KDTree, distances
from .atom import Atom
from .data import LatticeData, DataMap
from .shape import AbstractShape, Shape
from .basis import basis_t
from .structure import LatticeStructure

__all__ = ["Lattice"]

logger = logging.getLogger(__name__)


def _filter_dangling(indices, positions, neighbors, dists, min_neighbors):
    num_neighbors = np.count_nonzero(np.isfinite(dists), axis=1)
    sites = np.where(num_neighbors < min_neighbors)[0]
    if len(sites) == 0:
        return indices, positions, neighbors, dists
    elif len(sites) == indices.shape[0]:
        raise ValueError("Filtering min_neighbors would result in no sites!")

    # store current invalid index
    invalid_idx = indices.shape[0]

    # Remove data from arrays
    indices = np.delete(indices, sites, axis=0)
    positions = np.delete(positions, sites, axis=0)
    neighbors = np.delete(neighbors, sites, axis=0)
    dists = np.delete(dists, sites, axis=0)

    # Update neighbor indices and distances:
    # For each removed site below the neighbor index has to be decremented once
    mask = np.isin(neighbors, sites)
    neighbors[mask] = invalid_idx
    dists[mask] = np.inf
    for count, i in enumerate(sorted(sites)):
        neighbors[neighbors > (i - count)] -= 1

    # Update invalid indices in neighbor array since number of sites changed
    num_sites = indices.shape[0]
    neighbors[neighbors == invalid_idx] = num_sites

    return indices, positions, neighbors, dists


class Lattice(LatticeStructure):
    """Main lattice object representing a Bravais lattice model.

    Combines the ``LatticeBasis`` and the ``LatticeStructure`` class and adds
    the ability to construct finite lattice models.

    .. rubric:: Inheritance

    .. inheritance-diagram:: Lattice
       :parts: 1


    Parameters
    ----------
    basis: array_like or float or LatticeBasis
        The primitive basis vectors that define the unit cell of the lattice. If a
        ``LatticeBasis`` instance is passed it is copied and used as the new basis
        of the lattice.
    **kwargs
        Key-word arguments. Used for quickly configuring a ``Lattice`` instance.
        Allowed keywords are:

        Properties:
        atoms: Dictionary containing the atoms to add to the lattice.
        cons: Dictionary containing the connections to add to the lattice.
        shape: int or tuple defining the shape of the finite size lattice to build.
        periodic: int or list defining the periodic axes to set up.

    Examples
    --------
    Two dimensional lattice with one atom in the unit cell and nearest neighbors

    >>> import lattpy as lp
    >>> latt = lp.Lattice(np.eye(2))
    >>> latt.add_atom()
    >>> latt.add_connections(1)
    >>> _ = latt.build((5, 3))
    >>> latt
    Lattice(dim: 2, num_base: 1, num_neighbors: [4], shape: [5. 3.])

    Quick-setup of the same lattice:

    >>> import lattpy as lp
    >>> import matplotlib.pyplot as plt
    >>> latt = lp.Lattice.square(atoms={(0.0, 0.0): "A"}, cons={("A", "A"): 1})
    >>> _ = latt.build((5, 3))
    >>> _ = latt.plot()
    >>> plt.show()

    """

    def __init__(self, basis: basis_t, **kwargs):
        super().__init__(basis, **kwargs)

        # Lattice Cache
        self.data = LatticeData()
        self.shape = None
        self.pos = None
        self.periodic_axes = list()

        if "shape" in kwargs:
            self.build(kwargs["shape"], periodic=kwargs.get("periodic", None))

    @property
    def num_sites(self) -> int:
        """int: Number of sites in lattice data (if lattice has been built)."""
        return self.data.num_sites

    @property
    def num_cells(self) -> int:
        """int: Number of unit-cells in lattice data (if lattice has been built)."""
        return np.unique(self.data.indices[:, :-1], axis=0).shape[0]

    @property
    def indices(self):
        """np.ndarray: The lattice indices of the cached lattice data."""
        return self.data.indices

    @property
    def positions(self):
        """np.ndarray: The lattice positions of the cached lattice data."""
        return self.data.positions

    def volume(self) -> float:
        """The total volume (number of cells x cell-volume) of the built lattice.

        Returns
        -------
        vol : float
            The volume of the finite lattice structure.
        """
        return self.cell_volume * np.unique(self.data.indices[:, :-1], axis=0).shape[0]

    def alpha(self, idx: int) -> int:
        """Returns the atom component of the lattice index for a site in the lattice.

        Parameters
        ----------
        idx : int
            The super-index of a site in the cached lattice data.

        Returns
        -------
        alpha : int
            The index of the atom in the unit cell.
        """
        return self.data.indices[idx, -1]

    def atom(self, idx: int) -> Atom:
        """Returns the atom of a given site in the cached lattice data.

        Parameters
        ----------
        idx : int
            The super-index of a site in the cached lattice data.

        Returns
        -------
        atom : Atom
        """
        return self._atoms[self.data.indices[idx, -1]]

    def position(self, idx: int) -> np.ndarray:
        """Returns the position of a given site in the cached lattice data.

        Parameters
        ----------
        idx : int
            The super-index of a site in the cached lattice data.

        Returns
        -------
        pos : (D, ) np.ndarray
            The position of the lattice site.
        """
        return self.data.positions[idx]

    def index_from_position(self, pos: ArrayLike, atol: float = 1e-4) -> Optional[int]:
        """Returns the index of a given position.

        Parameters
        ----------
        pos : (D, ) array_like
            The position of the site in cartesian coordinates.
        atol : float, optional
            The absolute tolerance for comparing positions.

        Returns
        -------
        index : int or None
            The super-index of the site in the cached lattice data.
        """
        diff = self.data.positions - np.array(pos)[None, :]
        indices = np.where(np.all(np.abs(diff) < atol, axis=1))[0]
        if len(indices) == 0:
            return None
        return indices[0]

    def index_from_lattice_index(self, ind: ArrayLike) -> Optional[int]:
        """Returns the super-index of a site defined by the lattice index.

        Parameters
        ----------
        ind : (D + 1, ) array_like
            The lattice index ``(n_1, ..., n_D, alpha)`` of the site.

        Returns
        -------
        index : int or None
            The super-index of the site in the cached lattice data.
        """
        diff = self.data.indices - np.array(ind)[None, :]
        indices = np.where(np.all(np.abs(diff) < 1e-4, axis=1))[0]
        if len(indices) == 0:
            return None
        return indices[0]

    def neighbors(
        self, site: int, distidx: int = None, unique: bool = False
    ) -> np.ndarray:
        """Returns the neighours of a given site in the cached lattice data.

        Parameters
        ----------
        site : int
            The super-index of a site in the cached lattice data.
        distidx : int, optional
            Index of distance to the neighbors, default is 0 (nearest neighbors).
        unique : bool, optional
            If True, each unique pair is only returned once.

        Returns
        -------
        indices : np.ndarray of int
            The super-indices of the neighbors.
        """
        return self.data.get_neighbors(site, distidx, unique=unique)

    def nearest_neighbors(self, idx: int, unique: bool = False) -> np.ndarray:
        """Returns the nearest neighors of a given site in the cached lattice data.

        Parameters
        ----------
        idx : int
            The super-index of a site in the cached lattice data.
        unique : bool, optional
            If True, each unique pair is only return once.

        Returns
        -------
        indices : (N, ) np.ndarray of int
            The super-indices of the nearest neighbors.
        """
        return self.neighbors(idx, 0, unique)

    def iter_neighbors(
        self, site: int, unique: bool = False
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Iterates over the neighbors of all distances of a given site.

        Parameters
        ----------
        site : int
            The super-index of a site in the cached lattice data.
        unique : bool, optional
            If True, each unique pair is only return once.


        Yields
        ------
        distidx : int
            The distance index of the neighbor indices.
        neighbors : (N, ) np.ndarray
            The super-indices of the neighbors for the corresponding distance level.
        """
        return self.data.iter_neighbors(site, unique)

    def check_neighbors(self, idx0: int, idx1: int) -> Union[float, None]:
        """Checks if two sites are neighbors and returns the distance level if they are.

        Parameters
        ----------
        idx0 : int
            The first super-index of a site in the cached lattice data.
        idx1 : int
            The second super-index of a site in the cached lattice data.

        Returns
        -------
        distidx : int or None
            The distance index of the two sites if they are neighbors.
        """
        for distidx in range(self.num_distances):
            if idx1 in self.neighbors(idx0, distidx):
                return distidx
        return None

    def _update_shape(self):
        limits = self.data.get_limits()
        self.shape = limits[1] - limits[0]
        self.pos = limits[0]

    def build(
        self,
        shape: Union[float, Sequence[float], AbstractShape],
        primitive: bool = False,
        pos: Union[float, Sequence[float]] = None,
        check: bool = True,
        min_neighbors: int = None,
        num_jobs: int = -1,
        periodic: Union[bool, int, Sequence[int]] = None,
        callback: Callable = None,
        dtype: Union[int, str, np.dtype] = None,
        relative: bool = None,
    ):
        """Constructs the indices and neighbors of a finite size lattice.

        Parameters
        ----------
        shape : (N, ) array_like or float or AbstractShape
            Shape of finite size lattice to build.
        primitive : bool, optional
            If True the shape will be multiplied by the cell size of the model.
            The default is False.
        pos : (N, ) array_like or int, optional
            Optional position of the section to build. If ``None`` the origin is used.
        check : bool, optional
            If True the positions of the translation vectors are checked and
            filtered. The default is True. This should only be disabled if
            filtered later.
        min_neighbors : int, optional
            The minimum number of neighbors a site must have. This can be used to
            remove dangling sites at the edge of the lattice.
        num_jobs : int, optional
            Number of jobs to schedule for parallel processing of neighbors.
            If ``-1`` is given all processors are used. The default is ``-1``.
        periodic : int or array_like, optional
            Optional periodic axes to set. See ``set_periodic`` for mor details.
        callback : callable, optional
            The indices and positions are passed as arguments.
        dtype : int or str or np.dtype, optional
            Optional data-type for storing the lattice indices. Using a smaller
            bit-size may help reduce memory usage. By default, the given limits are
            checked to determine the smallest possible data-type.
        relative : bool, optional
            Same as ``primitive`` (backwards compatibility). Will be removed in a
            future version.

        Raises
        ------
        ValueError
            Raised if the dimension of the position doesn't match the dimension of
            the lattice.
        NoConnectionsError
            Raised if no connections have been set up.
        NotAnalyzedError
            Raised if the lattice distances and base-neighbors haven't been computed.
        """
        if relative is not None:
            warnings.warn(
                "``relative`` is deprecated and will be removed in a "
                "future version. Use ``primitive`` instead",
                DeprecationWarning,
            )
            primitive = relative

        self.data.reset()
        if not isinstance(shape, AbstractShape):
            basis = self.vectors if primitive else None
            shape = Shape(shape, pos=pos, basis=basis)
            # shape = np.atleast_1d(shape)

        self._assert_connections()
        self._assert_analyzed()

        logger.debug("Building lattice: %s at %s", shape, pos)

        # Build indices and positions
        indices, positions = self.build_indices(
            shape, primitive, pos, check, callback, dtype, True
        )

        # Compute the neighbors and distances between the sites
        neighbors, distances_ = self.compute_neighbors(indices, positions, num_jobs)
        if min_neighbors is not None:
            data = _filter_dangling(
                indices, positions, neighbors, distances_, min_neighbors
            )
            indices, positions, neighbors, distances_ = data

        # Set data of the lattice and update shape
        self.data.set(indices, positions, neighbors, distances_)
        self._update_shape()

        if periodic is not None:
            self.set_periodic(periodic, primitive)

        logger.debug(
            "Lattice shape: %s (%s)",
            self.shape,
            frmt_num(self.data.nbytes, unit="iB", div=1024),
        )
        return shape

    def _build_periodic_translation_vector(self, axes, primitive=False, indices=None):
        if indices is None:
            indices = self.indices.copy()

        axes = np.atleast_1d(axes)
        if not primitive:
            # Get lattice points limits
            indices = indices.copy()[:, :-1]  # strip alpha
            indices = np.unique(indices, axis=0)
            positions = self.transform(indices)
            limits = np.array([np.min(positions, axis=0), np.max(positions, axis=0)])
            shape = limits[1] - limits[0]
            # Get periodic point
            ppoint = np.zeros(self.dim)
            for ax in axes:
                ppoint[ax] = shape[ax] + self.cell_size[ax] * 2 / 3
            # Get periodic translation vector from point
            pnvec = self.itransform(ppoint)
            pnvec = np.round(pnvec, decimals=0)
        else:
            # Get index limits
            limits = np.array([np.min(indices, axis=0), np.max(indices, axis=0)])
            idx_size = (limits[1] - limits[0])[:-1]
            # Get periodic translation vector from limits
            pnvec = np.zeros_like(idx_size, dtype=np.int64)
            for ax in axes:
                pnvec[ax] = np.floor(idx_size[ax]) + 1
        return pnvec.astype(np.int64)

    def periodic_translation_vectors(self, axes, primitive=False):
        """Constrcuts all translation vectors for periodic boundary conditions.

        Parameters
        ----------
        axes : int or (N, ) array_like
            One or multiple axises to compute the translation vectors for.
        primitive : bool, optional
            Flag if the specified axes are in cartesian or lattice coordinates.
            If ``True`` the passed position will be multiplied with the lattice vectors.
            The default is ``False`` (cartesian coordinates).

        Returns
        -------
        nvecs : list of tuple
            The translation vectors for the periodic boundary conditions.
            The first item of each element is the axis, the second the
            corresponding translation vector.
        """
        # One axis: No combinations needed
        if isinstance(axes, int) or len(axes) == 1:
            return [(axes, self._build_periodic_translation_vector(axes, primitive))]
        # Add all combinations of the periodic axis
        items = list()
        for ax in itertools.combinations_with_replacement(axes, r=2):
            nvec = self._build_periodic_translation_vector(ax, primitive)
            items.append((ax, nvec))
            # Use +/- for every axis exept the first one to ensure all corners are hit
            if not np.all(np.array(ax) == axes[0]):
                nvec2 = np.copy(nvec)
                nvec2[1:] *= -1
                items.append((ax, nvec2))
        return items

    def _build_periodic(self, indices, positions, nvec, out_ind=None, out_pos=None):
        delta_pos = self.translate(nvec)
        delta_idx = np.append(nvec, 0)
        if out_ind is not None and out_pos is not None:
            out_ind[:] = indices + delta_idx
            out_pos[:] = positions + delta_pos
        else:
            out_ind = indices + delta_idx
            out_pos = positions + delta_pos
        return out_ind, out_pos

    def kdtree(self, positions=None, eps=0.0, boxsize=None):
        if positions is None:
            positions = self.data.positions
        k = np.sum(np.sum(self._raw_num_neighbors, axis=1)) + 1
        max_dist = np.max(self.distances) + 0.1 * np.min(self._raw_distance_matrix)
        return KDTree(positions, k, max_dist, eps=eps, boxsize=boxsize)

    def _compute_pneighbors(
        self, axis, primitive=False, indices=None, positions=None, num_jobs=-1
    ):
        if indices is None:
            indices = self.data.indices
            positions = self.data.positions

        axis = np.atleast_1d(axis)
        invald_idx = len(indices)

        # Build tree
        k = np.sum(np.sum(self._raw_num_neighbors, axis=1)) + 1
        max_dist = np.max(self.distances) + 0.1 * np.min(self._raw_distance_matrix)
        tree = KDTree(positions, k, max_dist)

        # Initialize arrays
        ind_t = np.zeros_like(indices)
        pos_t = np.zeros_like(positions)

        pidx, pdists, pnvecs, paxs = dict(), dict(), dict(), dict()
        for ax, nvec in self.periodic_translation_vectors(axis, primitive):
            # Translate positions along periodic axis
            self._build_periodic(indices, positions, nvec, ind_t, pos_t)

            # Query neighbors with translated points and filter
            neighbors, distances_ = tree.query(pos_t, num_jobs, self.DIST_DECIMALS)
            neighbors, distances_ = self._filter_neighbors(
                indices, neighbors, distances_, ind_t
            )

            # Convert to dict
            idx = np.where(np.isfinite(distances_).any(axis=1))[0]
            distances_ = distances_[idx]
            neighbors = neighbors[idx]
            for i, site in enumerate(idx):
                mask = i, neighbors[i] < invald_idx
                inds = neighbors[mask]
                dists = distances_[mask]
                # Update dict for indices `inds`
                pidx.setdefault(site, list()).extend(inds)  # noqa
                pdists.setdefault(site, list()).extend(dists)
                paxs.setdefault(site, list()).extend([ax] * len(inds))
                pnvecs.setdefault(site, list()).extend([nvec] * len(inds))
                # Update dict for neighbor indices of `inds`
                for j, d in zip(inds, dists):
                    pidx.setdefault(j, list()).append(site)  # noqa
                    pdists.setdefault(j, list()).append(d)
                    paxs.setdefault(j, list()).append(ax)
                    pnvecs.setdefault(j, list()).append(-nvec)

        # Convert values of dict to np.ndarray's
        for k in pidx.keys():
            vals, ind = np.unique(pidx[k], return_index=True)
            pidx[k] = np.array(vals)
            pdists[k] = np.array(pdists[k])[ind]
            paxs[k] = np.array(paxs[k])[ind]
            pnvecs[k] = np.array(pnvecs[k])[ind]
        return pidx, pdists, pnvecs, paxs

    def set_periodic(
        self, axis: Union[bool, int, Sequence[int]] = None, primitive: bool = False
    ):
        """Sets periodic boundary conditions along the given axis.

        Parameters
        ----------
        axis : bool or int or (N, ) array_like
            One or multiple axises to apply the periodic boundary conditions.
            If the axis is ``None`` the perodic boundary conditions will be removed.
        primitive : bool, optional
            Flag if the specified axes are in cartesian or lattice coordinates.
            If ``True`` the passed position will be multiplied with the lattice vectors.
            The default is ``False`` (cartesian coordinates).

        Raises
        ------
        NotBuiltError
            Raised if the lattice hasn't been built yet.

        Notes
        -----
        The lattice has to be built before applying the periodic boundarie conditions.
        The lattice also has to be at least three atoms big in the specified directions.
        """
        if isinstance(axis, bool):
            if axis is True:
                axis = np.arange(self.dim)
            else:
                axis = None

        logger.debug("Computing periodic neighbors along axis %s", axis)
        if self.shape is None:
            raise NotBuiltError()
        if axis is None:
            self.data.remove_periodic()
            self.periodic_axes = list()
        else:
            axis = np.atleast_1d(axis)
            pidx, pdists, pnvecs, paxs = self._compute_pneighbors(axis, primitive)
            self.data.set_periodic(pidx, pdists, pnvecs, paxs)
            self.periodic_axes = axis

    def _compute_connection_neighbors(self, positions1, positions2):
        # Set neighbor query parameters
        k = np.sum(np.sum(self._raw_num_neighbors, axis=1)) + 1
        max_dist = np.max(self.distances) + 0.1 * np.min(self._raw_distance_matrix)

        # Build sub-lattice tree's
        tree1 = KDTree(positions1, k=k, max_dist=max_dist)
        tree2 = KDTree(positions2, k=k, max_dist=max_dist)

        pairs = list()
        distances_ = list()
        # offset = len(positions1)
        connections = tree1.query_ball_tree(tree2, max_dist)
        for i, conns in enumerate(connections):
            if conns:
                conns = np.asarray(conns)
                dists = cdist(np.asarray([positions1[i]]), positions2[conns])[0]
                for j, dist in zip(conns, dists):
                    pairs.append((i, j))
                    # pairs.append((j, i))
                    distances_.append(dist)
                    # distances_.append(dist)

        return np.array(pairs), np.array(distances_)

    def compute_connections(self, latt):
        """Computes the connections between the current and another lattice.

        Parameters
        ----------
        latt : Lattice
            The other lattice.

        Returns
        -------
        neighbors : (N, 2) np.ndarray
            The connecting pairs between the two lattices.
            The first index of each row is the index in the current lattice data, the
            second one is the index for the other lattice ``latt``.
        distances : (N) np.ndarray
            The corresponding distances for the connections.
        """
        positions2 = latt.data.positions
        return self._compute_connection_neighbors(self.data.positions, positions2)

    def minimum_distances(self, site, primitive=False):
        """Computes the minimum distances between one site and the other lattice sites.

        This method can be used to find the distances in a lattice with
        periodic boundary conditions.

        Parameters
        ----------
        site : int
            The super-index i of a site in the cached lattice data.
        primitive : bool, optional
            Flag if the periopdic boundarey conditions are set up along cartesian or
            primitive basis vectors. The default is ``False`` (cartesian coordinates).

        Returns
        -------
        min_dists : (N, ) np.ndarray
            The minimum distances between the lattice site i and the other sites.
        """
        positions = self.positions
        # normal distances
        dists = [distances(positions[site], positions)]
        # periodic distances (to translated site)
        paxs = self.periodic_axes
        for axs, vec in self.periodic_translation_vectors(paxs, primitive):
            # Get position of translated lattice point and compute distances
            translated = self.translate(vec, positions[site])
            dists.append(distances(translated, positions))
            # reverse translate direction
            translated = self.translate(-vec, positions[site])
            dists.append(distances(translated, positions))
        # get minimum distances
        return np.min(dists, axis=0)

    def _append(
        self,
        ind,
        pos,
        neighbors,
        dists,
        ax=0,
        side=+1,
        sort_axis=None,
        sort_reverse=False,
        primitive=False,
    ):

        indices2 = np.copy(ind)
        positions2 = np.copy(pos)
        neighbors2 = np.copy(neighbors)
        distances2 = np.copy(dists)
        # Build translation vector
        indices = self.data.indices if side > 0 else indices2
        nvec = self._build_periodic_translation_vector(ax, primitive, indices)
        if side <= 0:
            nvec = -1 * nvec
        vec = self.translate(nvec)

        # Store temporary data
        positions1 = self.data.positions

        # Shift data of appended lattice
        indices2[:, :-1] += nvec
        positions2 += vec

        # Append data and compute connecting neighbors
        self.data.append(indices2, positions2, neighbors2, distances2)
        pairs, distances_ = self._compute_connection_neighbors(positions1, positions2)
        offset = len(positions1)
        for (i, j), dist in zip(pairs, distances_):
            self.data.add_neighbors(i, j + offset, dist)
            self.data.add_neighbors(j + offset, i, dist)

        if sort_axis is not None:
            self.data.sort(sort_axis, reverse=sort_reverse)

        # Update the shape of the lattice
        self._update_shape()

    # noinspection PyShadowingNames
    def append(
        self, latt, ax=0, side=+1, sort_ax=None, sort_reverse=False, primitive=False
    ):
        """Append another `Lattice`-instance along an axis.

        Parameters
        ----------
        latt : Lattice
            The other lattice to append to this instance.
        ax : int, optional
            The axis along the other lattice is appended. The default is 0 (x-axis).
        side : int, optional
            The side at which the new lattice is appended. If, for example, axis 0
            is used, the other lattice is appended on the right side if ``side=+1``
            and on the left side if ``side=-1``.
        sort_ax : int, optional
            The axis to sort the lattice indices after the other lattice has been
            added. The default is the value specified for ``ax``.
        sort_reverse : bool, optional
            If True, the lattice indices are sorted in reverse order.
        primitive : bool, optional
            Flag if the specified axes are in cartesian or lattice coordinates.
            If ``True`` the passed position will be multiplied with the lattice vectors.
            The default is ``False`` (cartesian coordinates).

        Examples
        --------
        >>> latt = Lattice(np.eye(2))
        >>> latt.add_atom(neighbors=1)
        >>> latt.build((5, 2))
        >>> latt.shape
        [5. 2.]

        >>> latt2 = Lattice(np.eye(2))
        >>> latt2.add_atom(neighbors=1)
        >>> latt2.build((2, 2))
        >>> latt2.shape
        [2. 2.]

        >>> latt.append(latt2, ax=0)
        >>> latt.shape
        [8. 2.]
        """
        ind = latt.data.indices
        pos = latt.data.positions
        neighbors = latt.data.neighbors
        dists = latt.data.distvals[latt.data.distances]
        self._append(
            ind, pos, neighbors, dists, ax, side, sort_ax, sort_reverse, primitive
        )

    def extend(self, size, ax=0, side=1, num_jobs=1, sort_ax=None, sort_reverse=False):
        """Extend the lattice along an axis.

        Parameters
        ----------
        size : float
            The size of which the lattice will be extended in direction of ``ax``.
        ax : int, optional
            The axis along the lattice is extended. The default is 0 (x-axis).
        side : int, optional
            The side at which the new lattice is appended. If, for example, axis 0
            is used, the lattice is extended to the right side if ``side=+1``
            and to the left side if ``side=-1``.
        num_jobs : int, optional
            Number of jobs to schedule for parallel processing of neighbors for new
            sites. If ``-1`` is given all processors are used. The default is ``-1``.
        sort_ax : int, optional
            The axis to sort the lattice indices after the lattice has been extended.
            The default is the value specified for ``ax``.
        sort_reverse : bool, optional
            If True, the lattice indices are sorted in reverse order.

        Examples
        --------
        >>> latt = Lattice(np.eye(2))
        >>> latt.add_atom(neighbors=1)
        >>> latt.build((5, 2))
        >>> latt.shape
        [5. 2.]

        >>> latt.extend(2, ax=0)
        [8. 2.]

        >>> latt.extend(2, ax=1)
        [8. 5.]
        """
        # Build indices and positions of new section
        shape = np.copy(self.shape)
        shape[ax] = size
        ind, pos = self.build_indices(shape, primitive=False, return_pos=True)
        # Compute the neighbors and distances between the sites of new section
        neighbors, dists = self.compute_neighbors(ind, pos, num_jobs)
        # Append new section
        self._append(ind, pos, neighbors, dists, ax, side, sort_ax, sort_reverse)

    def repeat(self, num=1, ax=0, side=1, sort_ax=None, sort_reverse=False):
        """Repeat the lattice along an axis.

        Parameters
        ----------
        num : int
            The number of times the lattice will be repeated in direction ``ax``.
        ax : int, optional
            The axis along the lattice is extended. The default is 0 (x-axis).
        side : int, optional
            The side at which the new lattice is appended. If, for example, axis 0
            is used, the lattice is extended to the right side if ``side=+1``
            and to the left side if ``side=-1``.
        sort_ax : int, optional
            The axis to sort the lattice indices after the lattice has been extended.
            The default is the value specified for ``ax``.
        sort_reverse : bool, optional
            If True, the lattice indices are sorted in reverse order.

        Examples
        --------
        >>> latt = Lattice(np.eye(2))
        >>> latt.add_atom(neighbors=1)
        >>> latt.build((5, 2))
        >>> latt.shape
        [5. 2.]

        >>> latt.repeat()
        [11.  2.]

        >>> latt.repeat(3)
        [35.  2.]

        >>> latt.repeat(ax=1)
        [35.  5.]
        """
        ind = self.data.indices
        pos = self.data.positions
        neighbors = self.data.neighbors
        dists = self.data.distvals[self.data.distances]
        for _ in range(num):
            self._append(ind, pos, neighbors, dists, ax, side, sort_ax, sort_reverse)

    def dmap(self) -> DataMap:
        """DataMap : Returns the data-map of the lattice model."""
        return self.data.map()

    def adjacency_matrix(self):
        """Computes the adjacency matrix for the neighbor data of the lattice.

        Returns
        -------
        adj_mat : (N, N) csr_matrix
            The adjacency matrix of the lattice.
        """
        num_sites = self.data.num_sites
        neighbors = self.data.neighbors
        dists = self.data.distances
        invalid_distidx = self.data.invalid_distidx
        # Build index pairs and corresponding distance array
        dtype = np.min_scalar_type(num_sites)
        sites = np.arange(num_sites, dtype=dtype)
        sites_t = np.tile(sites, (neighbors.shape[1], 1)).T
        pairs = np.reshape([sites_t, neighbors], newshape=(2, -1)).T
        distindices = dists.flatten()
        # Filter pairs with invalid indices
        mask = distindices != invalid_distidx
        pairs = pairs[mask]
        distindices = distindices[mask]

        rows, cols = pairs.T
        data = distindices + 1
        return csr_matrix((data, (rows, cols)), dtype=np.int8)

    # ==================================================================================

    def copy(self) -> "Lattice":
        """Lattice : Creates a (deep) copy of the lattice instance."""
        return deepcopy(self)

    def todict(self) -> dict:
        """Creates a dictionary containing the information of the lattice instance.

        Returns
        -------
        d : dict
            The information defining the current instance.
        """
        d = super().todict()
        d["shape"] = self.shape
        return d

    def dumps(self):  # pragma: no cover
        """Creates a string containing the information of the lattice instance.

        Returns
        -------
        s : str
            The information defining the current instance.
        """
        lines = list()
        for key, values in self.todict().items():
            head = key + ":"
            lines.append(f"{head:<15}" + "; ".join(str(x) for x in values))
        return "\n".join(lines)

    def dump(self, file: Union[str, int, bytes]) -> None:  # pragma: no cover
        """Save the data of the ``Lattice`` instance.

        Parameters
        ----------
        file : str or int or bytes
            File name to store the lattice. If ``None`` the hash of the lattice is used.
        """
        if file is None:
            file = f"{self.__hash__()}.latt"
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file: Union[str, int, bytes]) -> "Lattice":  # pragma: no cover
        """Load data of a saved ``Lattice`` instance.

        Parameters
        ----------
        file : str or int or bytes
            File name to load the lattice.

        Returns
        -------
        latt : Lattice
            The lattice restored from the file content.
        """
        with open(file, "rb") as f:
            latt = pickle.load(f)
        return latt

    def __hash__(self):
        import hashlib

        sha = hashlib.md5(self.dumps().encode("utf-8"))
        return int(sha.hexdigest(), 16)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def plot(
        self,
        lw: float = None,
        margins: Union[Sequence[float], float] = 0.1,
        legend: bool = None,
        grid: bool = False,
        pscale: float = 0.5,
        show_periodic: bool = True,
        show_indices: bool = False,
        index_offset: float = 0.1,
        con_colors: Sequence = None,
        adjustable: str = "box",
        ax: Union[plt.Axes, Axes3D] = None,
        show: bool = False,
    ) -> Union[plt.Axes, Axes3D]:  # pragma: no cover
        """Plot the cached lattice.

        Parameters
        ----------
        lw : float, optional
            Line width of the neighbor connections.
        margins : Sequence[float] or float, optional
            The margins of the plot.
        legend : bool, optional
            Flag if legend is shown
        grid : bool, optional
            If True, draw a grid in the plot.
        pscale : float, optional
            The scale for drawing periodic connections. The default is half of the
            normal length.
        show_periodic : bool, optional
            If True the periodic connections will be shown.
        show_indices : bool, optional
            If True the index of the sites will be shown.
        index_offset : float, optional
            The positional offset of the index text labels. Only used if
            `show_indices=True`.
        con_colors : Sequence[tuple], optional
            list of colors to override the defautl connection color. Each element
            has to be a tuple with the first two elements being the atom indices of
            the pair and the third element the color, for example ``[(0, 0, 'r')]``.
        adjustable : None or {'box', 'datalim'}, optional
            If not None, this defines which parameter will be adjusted to meet
            the equal aspect ratio. If 'box', change the physical dimensions of
            the Axes. If 'datalim', change the x or y data limits.
            Only applied to 2D plots.
        ax : plt.Axes or plt.Axes3D or None, optional
            Parent plot. If None, a new plot is initialized.
        show : bool, optional
            If True, show the resulting plot.
        """
        logger.debug("Plotting lattice")
        if self.dim > 3:
            raise ValueError(f"Plotting in {self.dim} dimensions is not supported!")

        hopz, atomz = range(2)

        fig, ax = subplot(self.dim, adjustable, ax=ax)
        # Draw sites
        for alpha in range(self.num_base):
            atom = self.atoms[alpha]
            col = atom.color or f"C{alpha}"
            points = self.data.get_positions(alpha)
            label = atom.name
            draw_sites(ax, points, atom.radius, color=col, label=label, zorder=atomz)
        # Draw connections
        ccolor = "k"
        pcolor = "0.5"
        positions = self.positions
        hop_colors = connection_color_array(self.num_base, ccolor, con_colors)
        per_colors = connection_color_array(self.num_base, pcolor)
        for i in range(self.num_sites):
            at1 = self.alpha(i)
            p1 = positions[i]
            for j in self.data.get_neighbors(i, periodic=False, unique=True):
                p2 = positions[j]
                at2 = self.alpha(j)
                color = hop_colors[at1][at2]
                draw_vectors(ax, p2 - p1, p1, color=color, lw=lw, zorder=hopz)
            if show_periodic:
                mask = self.data.neighbor_mask(i, periodic=True)
                idx = self.data.neighbors[i, mask]
                pnvecs = self.data.pnvecs[i, mask]
                neighbor_pos = self.data.positions[idx]
                for j, x in enumerate(neighbor_pos):
                    at2 = self.alpha(idx[j])
                    x = self.translate(-pnvecs[j], x)
                    color = per_colors[at1][at2]
                    vec = pscale * (x - p1)
                    draw_vectors(ax, vec, p1, color=color, lw=lw, zorder=hopz)
        # Add index labels
        if show_indices:
            positions = [self.position(i) for i in range(self.num_sites)]
            draw_indices(ax, positions, index_offset)
        # Configure legend
        if legend is None:
            legend = self.num_base > 1
        if legend:
            ax.legend()
        # Configure grid
        if grid and self.dim < 3:
            ax.set_axisbelow(True)
            ax.grid(b=True, which="major")
        # Adjust margin
        if isinstance(margins, float):
            margins = [margins] * self.dim
        ax.margins(*margins)

        fig.tight_layout()
        if show:
            plt.show()
        return ax

    def __repr__(self) -> str:
        shape = str(self.shape) if self.shape is not None else "None"
        return (
            f"{self.__class__.__name__}("
            f"dim: {self.dim}, "
            f"num_base: {self.num_base}, "
            f"num_neighbors: {self.num_neighbors}, "
            f"shape: {shape})"
        )
