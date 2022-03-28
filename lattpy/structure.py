# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones


# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Lattice structure object for defining the atom basis and neighbor connections."""

import pickle
import logging
import warnings
import itertools
import collections
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from typing import Union, Tuple, List, Sequence, Callable, Any, Dict
from .utils import (
    ArrayLike,
    SiteOccupiedError,
    NoAtomsError,
    NoConnectionsError,
    NotAnalyzedError,
)
from .plotting import (
    subplot,
    draw_sites,
    draw_vectors,
    draw_unit_cell,
    connection_color_array,
)
from .spatial import vindices, interweave, KDTree
from .atom import Atom
from .shape import AbstractShape
from .basis import basis_t, LatticeBasis

logger = logging.getLogger(__name__)

__all__ = ["LatticeStructure"]


class LatticeStructure(LatticeBasis):
    """Structure object representing a infinite Bravais lattice.

    Combines the ``LatticeBasis`` with a set of atoms and connections between
    them to define a general lattice structure.

    .. rubric:: Inheritance

    .. inheritance-diagram:: LatticeStructure
       :parts: 1


    Parameters
    ----------
    basis : array_like or float or LatticeBasis
        The primitive basis vectors that define the unit cell of the lattice. If a
        ``LatticeBasis`` instance is passed it is copied and used as the new basis
        of the lattice.
    **kwargs
        Key-word arguments. Used for quickly configuring a ``LatticeStructure``
        instance. Allowed keywords are:

        Properties:
        atoms: Dictionary containing the atoms to add to the lattice.
        cons: Dictionary containing the connections to add to the lattice.

    Examples
    --------
    Two dimensional lattice with one atom in the unit cell and nearest neighbors

    >>> import lattpy as lp
    >>> latt = lp.LatticeStructure(np.eye(2))
    >>> latt.add_atom()
    >>> latt.add_connections(1)
    >>> latt
    LatticeStructure(dim: 2, num_base: 1, num_neighbors: [4])

    Quick-setup of the same lattice:

    >>> import lattpy as lp
    >>> import matplotlib.pyplot as plt
    >>> latt = lp.LatticeStructure.square(atoms={(0.0, 0.0): "A"}, cons={("A", "A"): 1})
    >>> _ = latt.plot_cell()
    >>> plt.show()

    """

    # Decimals used for rounding distances
    DIST_DECIMALS: int = 6

    def __init__(self, basis: basis_t, **kwargs):
        super().__init__(basis)
        # Atom data
        self._num_base = 0
        self._atoms = list()
        self._positions = list()

        # Raw neighbor data without including connections
        self._raw_base_neighbors = None
        self._raw_distance_matrix = None
        self._raw_num_neighbors = None

        # Neighbour data
        self._connections = None
        self._base_neighbors = None
        self._num_neighbors = None
        self._distance_matrix = None
        self._distances = None

        logger.debug(
            "LatticeStructure initialized (D=%i)\nvectors:\n%s", self.dim, self._vectors
        )

        if "atoms" in kwargs:
            atom_dict = kwargs["atoms"]
            for pos, atom in atom_dict.items():
                self.add_atom(pos, atom)
        if "cons" in kwargs:
            cons = kwargs["cons"]
            if isinstance(cons, int):
                self.add_connections(cons)
            else:
                cons_dict = kwargs["cons"]
                for pair, num in cons_dict.items():
                    self.add_connection(*pair, num)
                self.analyze()

    @property
    def num_base(self) -> int:
        """int: The number of atoms in the unit cell."""
        return self._num_base

    @property
    def atoms(self) -> List[Atom]:
        """list of Atom: List of the atoms in the unit cell."""
        return self._atoms

    @property
    def atom_positions(self) -> List[np.ndarray]:
        """list of np.ndarray: List of positions of the atoms in the unit cell."""
        return self._positions

    @property
    def num_distances(self) -> int:
        """int: The maximal number of distances between the lattice sites."""
        return int(np.max(self._connections))

    @property
    def num_neighbors(self) -> np.ndarray:
        """np.ndarray: The number of neighbors of each atom in the unitcell."""
        return self._num_neighbors

    @property
    def base_neighbors(self) -> np.ndarray:
        """np.ndarray: The neighbors of the unitcell at the origin."""
        return self._base_neighbors

    @property
    def distances(self) -> np.ndarray:
        """np.ndarray: List of distances between the lattice sites."""
        return self._distances

    def add_atom(
        self,
        pos: Union[float, Sequence[float]] = None,
        atom: Union[str, Dict[str, Any], Atom] = None,
        primitive: bool = False,
        neighbors: int = 0,
        relative: bool = None,
        **kwargs,
    ) -> Atom:
        """Adds a site to the basis of the lattice unit cell.

        Parameters
        ----------
        pos : (N) array_like or float, optional
            Position of site in the unit-cell. The default is the origin of the cell.
            The size of the array has to match the dimension of the lattice.
        atom : str or dict or Atom, optional
            Identifier of the site. If a string is passed, a new Atom instance is
            created.
        primitive : bool, optional
            Flag if the specified position is in cartesian or lattice coordinates.
            If True the passed position will be multiplied with the lattice vectors.
            The default is ``False`` (cartesian coordinates).
        neighbors : int, optional
            The number of neighbor distance to calculate. If the number is 0 the
            distances have to be calculated manually after configuring the
            lattice basis.
        relative : bool, optional
            Same as ``primitive`` (backwards compatibility). Will be removed in a
            future version.
        **kwargs
            Keyword arguments for ´Atom´ constructor. Only used if a new ``Atom``
            instance is created.

        Returns
        -------
        atom : Atom
            The ``Atom``-instance of the newly added site.

        Raises
        ------
        ValueError
            Raised if the dimension of the position does not match the dimension
            of the lattice.
        ConfigurationError
            Raised if the position of the new atom is already occupied.

        Examples
        --------
        Construct a square lattice

        >>> latt = LatticeStructure(np.eye(2))

        Create an atom and add it to the origin of the unit cell of the lattice

        >>> atom1 = Atom(name="A")
        >>> latt.add_atom([0.0, 0.0], atom=atom1)
        >>> latt.get_atom(0)
        Atom(A, size=10, 0)

        An ``Atom`` instance can also be created by passing the name of the (new) atom
        and optional keyword arguments for the constructor:

        >>> latt.add_atom([0.5, 0.5], atom="B", size=15)
        >>> latt.get_atom(1)
        Atom(B, size=15, 1)
        """
        if relative is not None:
            warnings.warn(
                "``relative`` is deprecated and will be removed in a "
                "future version. Use ``primitive`` instead",
                DeprecationWarning,
            )
            primitive = relative

        pos = np.zeros(self.dim) if pos is None else np.atleast_1d(pos)
        if primitive:
            pos = self.translate(pos)

        if len(pos) != self._dim:
            raise ValueError(
                f"Shape of the position {pos} doesn't match "
                f"the dimension {self.dim} of the lattice!"
            )
        if any(np.all(pos == x) for x in self._positions):
            raise SiteOccupiedError(atom, pos)

        if not isinstance(atom, Atom):
            atom = Atom(atom, **kwargs)

        self._atoms.append(atom)
        self._positions.append(np.asarray(pos))

        # Update number of base atoms if data is valid
        num_base = len(self._atoms)
        assert num_base == len(self._positions)
        self._num_base = num_base
        logger.debug("Added atom %s at %s", atom, pos)

        # Initial array for number of neighbour distances
        # for current number of atoms in the unit cell
        self._connections = np.zeros((num_base, num_base), dtype=np.int64)

        if neighbors:
            self.add_connections(neighbors)
        return atom

    def get_alpha(self, atom: Union[int, str, Atom]) -> Union[int, List[int]]:
        """Returns the index of the atom in the unit-cell.

        Parameters
        ----------
        atom : int or str or Atom
            The argument for getting the atom. If a ``int`` is passed
            it is interpreted as the index, if a ``str`` is passed as
            the name of an atom.

        Returns
        -------
        alpha : int or list of int
            The atom indices. If a string was passed multiple atoms with the same name
            can be returned as list.

        Examples
        --------
        Construct a lattice with two identical atoms and a third atom in the unit cell:

        >>> latt = LatticeStructure(np.eye(2))
        >>> latt.add_atom([0, 0], atom="A")
        >>> latt.add_atom([0.5, 0], atom="B")
        >>> latt.add_atom([0.5, 0.5], atom="B")

        Get the atom index of atom A:

        >>> latt.get_alpha("A")
        [0]

        Get the indices of the atoms B:

        >>> latt.get_alpha("B")
        [1, 2]

        Since there are two atoms B in the unit cell of the lattice both indices
        are returned.
        """
        if isinstance(atom, Atom):
            return self._atoms.index(atom)
        elif isinstance(atom, str):
            return [i for i, at in enumerate(self._atoms) if atom == at.name]
        return atom

    def get_atom(self, atom: Union[int, str, Atom]) -> Atom:
        """Returns the ``Atom`` instance of the given atom in the unit cell.

        Parameters
        ----------
        atom : int or str or Atom
            The argument for getting the atom. If a ``int`` is passed
            it is interpreted as the index, if a ``str`` is passed as
            the name of an atom.

        Returns
        -------
        atom : Atom
            The ``Atom`` instance of the given site.

        See Also
        --------
        get_alpha : Get the index of the given atom.

        Examples
        --------
        Construct a lattice with one atom in the unit cell

        >>> latt = LatticeStructure(np.eye(2))
        >>> latt.add_atom([0, 0], atom="A")

        Get the atom instance by the name

        >>> latt.get_atom("A")
        Atom(A, size=10, 0)

        or by the index:

        >>> latt.get_atom(0)
        Atom(A, size=10, 0)

        """
        if isinstance(atom, Atom):
            return atom
        elif isinstance(atom, int):
            return self._atoms[atom]
        else:
            for at in self._atoms:
                if atom == at.name:
                    return at
            raise ValueError(f"No Atom with the name '{atom}' found!")

    def add_connection(
        self,
        atom1: Union[int, str, Atom],
        atom2: Union[int, str, Atom],
        num_distances=1,
        analyze: bool = False,
    ) -> None:
        """Sets the number of distances for a specific connection between two atoms.

        Parameters
        ----------
        atom1 : int or str or Atom
            The first atom of the connected pair.
        atom2 : int or str or Atom
            The second atom of the connected pair.
        num_distances : int, optional
            The number of neighbor-distance levels, e.g. setting to ``1`` means
            only nearest neighbors. The default are nearest neighbor connections.
        analyze : bool, optional
            If True the lattice basis is analyzed after adding connections.
            If ``False`` the ``analyze``-method needs to be called manually.
            The default is ``False``.

        See Also
        --------
        add_connections : Set up connections for all atoms in the unit cell in one call.
        analyze : Called after setting up all the lattice connections.

        Examples
        --------
        Construct a square lattice with two atoms, A and B, in the unit cell:

        >>> latt = LatticeStructure(np.eye(2))
        >>> latt.add_atom([0.0, 0.0], atom="A")
        >>> latt.add_atom([0.5, 0.5], atom="B")

        Set next nearest and nearest neighbors between the A atoms:

        >>> latt.add_connection("A", "A", num_distances=2)

        Set nearest neighbors between A and B:

        >>> latt.add_connection("A", "B", num_distances=1)

        Set nearest neighbors between the B atoms:

        >>> latt.add_connection("B", "B", num_distances=1)
        """
        alpha1 = np.atleast_1d(self.get_alpha(atom1))
        alpha2 = np.atleast_1d(self.get_alpha(atom2))
        for i, j in itertools.product(alpha1, alpha2):
            self._connections[i, j] = num_distances
            self._connections[j, i] = num_distances
        if analyze:
            self.analyze()

    def add_connections(self, num_distances=1, analyze: bool = True) -> None:
        """Sets the number of distances for all possible atom-pairs of the unitcell.

        Parameters
        ----------
        num_distances : int, optional
            The number of neighbor-distance levels, e.g. setting to ``1`` means
            only nearest neighbors. The default are nearest neighbor connections.
        analyze : bool, optional
            If True the lattice basis is analyzed after adding connections.
            If ``False`` the ``analyze``-method needs to be called manually.
            The default is True.

        See Also
        --------
        add_connection : Set up connection between two specific atoms in the unit cell.
        analyze : Called after setting up all the lattice connections.

        Examples
        --------
        Construct a square lattice with one atom in the unit cell:

        >>> latt = LatticeStructure(np.eye(2))
        >>> latt.add_atom()

        Set nearest neighbor hopping:

        >>> latt.add_connections(num_distances=1)
        """
        self._connections.fill(num_distances)
        if analyze:
            self.analyze()

    def set_num_neighbors(self, num_neighbors: int = 1, analyze: bool = True) -> None:
        """Sets the maximal neighbor distance of the lattice.

        Parameters
        ----------
        num_neighbors: int, optional
            The number of neighbor-distance levels,
            e.g. setting to ``1`` means only nearest neighbors.
        analyze: bool
            Flag if lattice base is analyzed. If ``False`` the ``analyze``-method
            needs to be called manually. The default is True.
        """
        warnings.warn(
            "Configuring neighbors with `set_num_neighbors` is deprecated "
            "and will be removed in a future version. Use the "
            "`add_connections` instead.",
            DeprecationWarning,
        )
        self.add_connections(num_neighbors, analyze)

    def _compute_base_neighbors(self, max_distidx, num_jobs=1):
        logger.debug("Building indices of neighbor-cells")

        # Build indices of neighbor-cells
        self._positions = np.asarray(self._positions)
        cell_range = 2 * max_distidx
        logger.debug("Max. distidx: %i, Cell-range: %i", max_distidx, cell_range)

        nvecs = self.get_neighbor_cells(
            cell_range, include_origin=True, comparison=np.less_equal
        )
        arrays = [
            np.c_[nvecs, i * np.ones(nvecs.shape[0])] for i in range(self.num_base)
        ]
        cols = self.dim + 1
        indices = np.ravel(arrays, order="F").astype(np.int64)
        indices = indices.reshape(cols, int(indices.shape[0] / cols)).T

        # Compute positions and initialize tree
        nvecs, alphas = indices[:, :-1], indices[:, -1]
        positions = self.translate(nvecs, np.array(self.atom_positions)[alphas])
        tree = KDTree(positions, k=len(positions))

        # Compute neighbors for each distance level in the cell range
        n = self.num_base
        neighbor_array = [
            [collections.OrderedDict() for _ in range(n)] for _ in range(n)
        ]
        for a1 in range(self.num_base):
            pos = self.atom_positions[a1]
            neighbors, distances = tree.query(
                pos, num_jobs, self.DIST_DECIMALS, include_zero=True, compact=False
            )
            neighbor_indices = indices[neighbors]
            # Store neighbors of certain distance for each atom pair in the unit cell
            for dist, idx in zip(distances, neighbor_indices):
                a2 = idx[-1]
                if dist:
                    neighbor_array[a1][a2].setdefault(dist, list()).append(idx)

        # Remove extra neighbors
        for a1, a2 in itertools.product(range(n), repeat=2):
            neighbors = neighbor_array[a1][a2]
            dists = list(sorted(neighbors.keys()))
            for dist in dists[:max_distidx]:
                neighbor_array[a1][a2][dist] = np.array(neighbors[dist])
            for dist in dists[max_distidx:]:
                del neighbor_array[a1][a2][dist]

        return neighbor_array

    def _analyze_raw(self, max_distidx):
        """Analyzes the structure of the raw lattice (without connections)."""
        n = self.num_base
        # Compute raw neighbors of unit cell
        neighbor_array = self._compute_base_neighbors(max_distidx)

        # Compute the raw distance matrix and the raw number of neighbors
        raw_distance_matrix = [[list() for _ in range(n)] for _ in range(n)]
        raw_num_neighbors = np.zeros((n, n), dtype=np.int64)
        for a1, a2 in itertools.product(range(n), repeat=2):
            neighbors = neighbor_array[a1][a2]
            raw_distance_matrix[a1][a2] += list(neighbors.keys())
            raw_num_neighbors[a1, a2] = sum(len(x) for x in neighbors.values())

        # Save raw neighbor data of the unit cell
        self._raw_base_neighbors = neighbor_array
        self._raw_distance_matrix = raw_distance_matrix
        self._raw_num_neighbors = raw_num_neighbors
        logger.debug("Number of raw neighbors:\n%s", raw_num_neighbors)
        logger.debug("Raw distance-matrix:\n%s", raw_distance_matrix)

    def _assert_atoms(self):
        if len(self._atoms) == 0:
            raise NoAtomsError()

    def _assert_connections(self):
        if np.all(self._connections == 0):
            raise NoConnectionsError()

    def _assert_analyzed(self):
        if not self._base_neighbors:
            raise NotAnalyzedError()

    def analyze(self) -> None:
        """Analyzes the structure of the lattice and stores neighbor data.

        Check's distances between all sites of the bravais lattice and saves the ``n``
        lowest values. The neighbor lattice-indices of the unit-cell are also stored
        for later use. This speeds up many calculations like finding nearest neighbors.

        Raises
        ------
        NoAtomsError
            Raised if no atoms where added to the lattice. The atoms in the unit cell
            are needed for computing the neighbors and distances of the lattice.
        NoConnectionsError
            Raised if no connections have been set up.

        Notes
        -----
        Before calling the ``analyze`` function all connections in the lattice have to
        be set up.

        Examples
        --------
        Construct a square lattice with one atom in the unit cell and nearest neighbors:

        >>> latt = LatticeStructure(np.eye(2))
        >>> latt.add_atom()
        >>> latt.add_connections(num_distances=1)

        Call `analyze` after setting up the connections

        >>> latt.analyze()
        """
        logger.debug("Analyzing lattice")

        self._assert_atoms()
        self._assert_connections()

        max_distidx = int(np.max(self._connections))
        n = self.num_base

        # Analyze the raw lattice
        self._analyze_raw(max_distidx)

        # Filter base neighbor data for configured connections and
        # store neighbors and distances as list for each atom
        base_neighbors = [collections.OrderedDict() for _ in range(n)]
        base_distance_matrix = [[list() for _ in range(n)] for _ in range(n)]
        unique_distances = set()
        for a1, a2 in itertools.product(range(n), repeat=2):
            neighbors = self._raw_base_neighbors[a1][a2]
            dists = list(neighbors.keys())
            max_dist = self._connections[a1, a2]
            for distidx, dist in enumerate(dists[:max_dist]):
                unique_distances.add(dist)
                base_neighbors[a1].setdefault(dist, list()).extend(neighbors[dist])
                base_distance_matrix[a1][a2].append(dist)
            base_distance_matrix[a1][a2] = list(sorted(base_distance_matrix[a1][a2]))

        # Convert base neighbors back to np.ndarray
        for a1 in range(self.num_base):
            for key, vals in base_neighbors[a1].items():
                base_neighbors[a1][key] = np.asarray(vals)

        max_num_distances = len(unique_distances)

        # Compute number of neighbors for each atom in the unit cell
        num_neighbors = np.zeros(self.num_base, dtype=np.int8)
        for i, neighbors in enumerate(base_neighbors):
            num_neighbors[i] = sum(len(indices) for indices in neighbors.values())

        # store distance values / keys:
        distances = np.zeros((self.num_base, max_num_distances))
        for alpha in range(self.num_base):
            try:
                dists = list(base_neighbors[alpha].keys())
            except ValueError:
                dists = list()
            distances[alpha, : len(dists)] = sorted(dists)

        self._base_neighbors = base_neighbors
        self._distance_matrix = base_distance_matrix
        self._num_neighbors = num_neighbors
        self._distances = distances
        logger.debug("Number of neighbors:\n%s", num_neighbors)
        logger.debug("Distance-matrix:\n%s", base_distance_matrix)
        logger.debug("Distances:\n%s", distances)

    def get_position(
        self, nvec: Union[int, Sequence[int]] = None, alpha: int = 0
    ) -> np.ndarray:
        """Returns the position for a given translation vector ``nvec`` and atom
        ``alpha``.

        Parameters
        ----------
        nvec : (N) array_like or int
            The translation vector.
        alpha : int, optional
            The atom index, default is 0.

        Returns
        -------
        pos : (N) np.ndarray
            The position of the transformed lattice site.

        Examples
        --------
        >>> latt = LatticeStructure(np.eye(2))
        >>> latt.add_atom()
        >>> latt.add_connections(1, analyze=True)
        >>> latt.get_position([1, 0], alpha=0)
        [1. 0.]
        """
        r = self._positions[alpha]
        if nvec is None:
            return r
        n = np.atleast_1d(nvec)
        return r + (self._vectors @ n)  # self.translate(n, r)

    def get_positions(self, indices):
        """Returns the positions for multiple lattice indices.

        Parameters
        ----------
        indices : (N, D+1) array_like or int
            List of lattice indices in the format :math:`(n_1, ..., n_d, α)`.

        Returns
        -------
        pos : (N, D) np.ndarray
            The positions of the lattice sites.

        Examples
        --------
        >>> latt = LatticeStructure(np.eye(2))
        >>> latt.add_atom()
        >>> latt.add_connections(1)
        >>> ind = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
        >>> latt.get_positions(ind)
        [[0. 0.]
         [1. 0.]
         [1. 1.]]
        """
        indices = np.asarray(indices)
        nvecs, alphas = indices[:, :-1], indices[:, -1]
        return self.translate(nvecs, np.array(self.atom_positions)[alphas])

    def estimate_index(self, pos: Union[float, Sequence[float]]) -> np.ndarray:
        """Returns the nearest matching lattice index (n, alpha) for a position.

        Parameters
        ----------
        pos : array_like or float
            The position of the site in world coordinates.

        Returns
        -------
        nvec : np.ndarray
            The estimated translation vector :math:`n`.

        Examples
        --------
        >>> latt = LatticeStructure(np.eye(2))
        >>> latt.add_atom()
        >>> latt.add_connections(1)
        >>> latt.estimate_index([1.2, 0.2])
        [1 0]
        """
        pos = np.asarray(pos)
        n = np.asarray(np.round(self._vectors_inv @ pos, decimals=0), dtype="int")
        return n

    def get_neighbors(
        self, nvec: Union[int, Sequence[int]] = None, alpha: int = 0, distidx: int = 0
    ) -> np.ndarray:
        """Returns the neighour indices of a given site by transforming neighbor data.

        Parameters
        ----------
        nvec: (D) array_like or int, optional
            The translation vector, the default is the origin.
        alpha: int, optional
            The atom index, default is 0.
        distidx: int, optional
            The index of distance to the neighbors, default is 0 (nearest neighbors).

        Returns
        -------
        indices: (N, D+1) np.ndarray
            The lattice indices of the neighbor sites.

        Raises
        ------
        NotAnalyzedError
            Raised if the lattice distances and base-neighbors haven't been computed.

        Examples
        --------
        >>> latt = LatticeStructure(np.eye(2))
        >>> latt.add_atom()
        >>> latt.add_connections(1)
        >>> latt.get_neighbors(nvec=[0, 0], alpha=0, distidx=0)
        [[ 1  0  0]
         [ 0 -1  0]
         [-1  0  0]
         [ 0  1  0]]
        """
        if nvec is None:
            nvec = np.zeros(self.dim)
        self._assert_analyzed()
        logger.debug(
            "Computing neighbor-indices of %s, %i (distidx: %i)", nvec, alpha, distidx
        )

        nvec = np.atleast_1d(nvec)
        keys = list(sorted(self._base_neighbors[alpha].keys()))
        dist = keys[distidx]
        indices = self._base_neighbors[alpha][dist]
        indices_transformed = indices.copy()
        indices_transformed[:, :-1] += nvec.astype(np.int64)
        logger.debug("Neighbour-indices: %s", indices_transformed)

        return indices_transformed

    def get_neighbor_positions(
        self, nvec: Union[int, Sequence[int]] = None, alpha: int = 0, distidx: int = 0
    ) -> np.ndarray:
        """Returns the neighour positions of a given site by transforming neighbor data.

        Parameters
        ----------
        nvec: (D) array_like or int, optional
            The translation vector, the default is the origin.
        alpha: int, optional
           The site index, default is 0.
        distidx: int, default
            The index of distance to the neighbors, default is 0 (nearest neighbors).

        Returns
        -------
        positions: (N, D) np.ndarray
            The positions of the neighbor sites.

        Raises
        ------
        NotAnalyzedError
            Raised if the lattice distances and base-neighbors haven't been computed.

        Examples
        --------
        >>> latt = LatticeStructure(np.eye(2))
        >>> latt.add_atom()
        >>> latt.add_connections(1)
        >>> latt.get_neighbor_positions(nvec=[0, 0], alpha=0, distidx=0)
        [[ 1.  0.]
         [ 0. -1.]
         [-1.  0.]
         [ 0.  1.]]
        """
        if nvec is None:
            nvec = np.zeros(self.dim)
        self._assert_analyzed()
        logger.debug(
            "Computing neighbor-positions of %s, %i (distidx: %i)", nvec, alpha, distidx
        )

        indices = self.get_neighbors(nvec, alpha, distidx)
        nvecs, alphas = indices[:, :-1], indices[:, -1]
        atom_pos = self._positions[alphas]
        positions = self.translate(nvecs, atom_pos)
        logger.debug("Neighbour-positions: %s", positions)

        return positions

    def get_neighbor_vectors(
        self, alpha: int = 0, distidx: int = 0, include_zero: bool = False
    ) -> np.ndarray:
        """Returns the vectors to the neigbor sites of an atom in the unit cell.

        Parameters
        ----------
        alpha : int, optional
            Index of the base atom. The default is the first atom in the unit cell.
        distidx : int, default
            Index of distance to the neighbors, default is 0 (nearest neighbors).
        include_zero : bool, optional
            Flag if zero-vector is included in result. The default is ``False``.

        Returns
        -------
        vectors : np.ndarray
            The vectors from the site of the atom :math:`alpha` to the neighbor sites.

        Raises
        ------
        NotAnalyzedError
            Raised if the lattice distances and base-neighbors haven't been computed.

        Examples
        --------
        >>> latt = LatticeStructure(np.eye(2))
        >>> latt.add_atom()
        >>> latt.add_connections(1)
        >>> latt.get_neighbor_vectors(alpha=0, distidx=0)
        [[ 1.  0.]
         [ 0. -1.]
         [-1.  0.]
         [ 0.  1.]]
        """
        self._assert_analyzed()
        logger.debug(
            "Computing neighbor-vectors of atom %i (distidx: %i)", alpha, distidx
        )

        pos0 = self._positions[alpha]
        pos1 = self.get_neighbor_positions(alpha=alpha, distidx=distidx)
        if include_zero:
            pos1 = np.append(np.zeros((1, self.dim)), pos1, axis=0)
        vecs = pos1 - pos0
        logger.debug("Neighbour-vectors: %s", vecs)

        return vecs

    def fourier_weights(
        self, k: ArrayLike, alpha: int = 0, distidx: int = 0
    ) -> np.ndarray:
        """Returns the Fourier-weight for a given vector.

        Parameters
        ----------
        k : array_like
            The wavevector to compute the lattice Fourier-weights.
        alpha : int, optional
            Index of the base atom. The default is the first atom in the unit cell.
        distidx : int, default
            Index of distance to the neighbors, default is 0 (nearest neighbors).

        Returns
        -------
        weight : np.ndarray
        """
        vecs = self.get_neighbor_vectors(alpha=alpha, distidx=distidx)
        # weights = np.sum([np.exp(1j * np.dot(k, v)) for v in vecs])
        weights = np.sum(np.exp(1j * np.inner(k, vecs)))
        return weights

    def get_base_atom_dict(
        self, atleast2d: bool = True
    ) -> Dict[Any, List[Union[np.ndarray, Any]]]:
        """Returns a dictionary containing the positions for eatch of the base atoms.

        Parameters
        ----------
        atleast2d : bool, optional
            If True, one-dimensional coordinates will be cast to 2D vectors.

        Returns
        -------
        atom_pos : dict
            The positions of the atoms as a dictionary.

        Examples
        --------
        >>> latt = LatticeStructure(np.eye(2))
        >>> latt.add_atom([0, 0], atom="A")
        >>> latt.add_atom([0.5, 0], atom="B")
        >>> latt.add_atom([0.5, 0.5], atom="B")
        >>> latt.get_base_atom_dict()
        {
            Atom(A, radius=0.2, 0): [array([0, 0])],
            Atom(B, radius=0.2, 1): [array([0.5, 0. ]), array([0.5, 0.5])]
        }
        """
        atom_pos = dict()
        for atom, pos in zip(self._atoms, self._positions):
            if atleast2d and self.dim == 1:
                pos = np.array([pos, 0])

            if atom.name in atom_pos.keys():
                atom_pos[atom].append(pos)
            else:
                atom_pos[atom] = [pos]
        return atom_pos

    # noinspection PyShadowingNames
    def check_points(
        self,
        points: np.ndarray,
        shape: Union[int, Sequence[int], AbstractShape],
        relative: bool = False,
        pos: Union[float, Sequence[float]] = None,
        tol: float = 1e-3,
    ) -> np.ndarray:
        """Returns a mask for the points in the given shape.

        Parameters
        ----------
        points: (M, N) np.ndarray
            The points in cartesian coordinates.
        shape: (N) array_like or int or AbstractShape
            shape of finite size lattice to build.
        relative: bool, optional
            If True the shape will be multiplied by the cell size of the model.
            The default is True.
        pos: (N) array_like or int, optional
            Optional position of the section to build. If ``None`` the origin is used.
        tol: float, optional
            The tolerance for checking the points. The default is ``1e-3``.

        Returns
        -------
        mask: (M) np.ndarray
            The mask for the points inside the shape.

        Examples
        --------
        >>> latt = LatticeStructure(np.eye(2))
        >>> shape = (2, 2)
        >>> points = np.array([[0, 0], [2, 2], [3, 2]])
        >>> latt.check_points(points, shape)
        [ True  True False]
        """
        if isinstance(shape, AbstractShape):
            return shape.contains(points, tol)
        else:
            shape = np.atleast_1d(shape)
            if len(shape) != self.dim:
                raise ValueError(
                    f"Dimension of shape {len(shape)} doesn't "
                    f"match the dimension of the lattice {self.dim}"
                )
            if relative:
                shape += np.max(self.vectors, axis=0) - 0.1 * self.norms

            pos = np.zeros(self.dim) if pos is None else np.array(pos, dtype=np.float64)
            pos -= tol
            end = pos + shape + tol

            mask = (pos[0] <= points[:, 0]) & (points[:, 0] <= end[0])
            for i in range(1, self.dim):
                mask = mask & (pos[i] <= points[:, i]) & (points[:, i] <= end[i])
            return mask

    def build_translation_vectors(
        self,
        shape: Union[int, Sequence[int], AbstractShape],
        primitive: bool = False,
        pos: Union[float, Sequence[float]] = None,
        check: bool = True,
        dtype: Union[int, np.dtype] = None,
        oversample: float = 0.0,
    ) -> np.ndarray:
        """Constructs the translation vectors :math:`n` in a given shape.

        Raises
        ------
        ValueError
            Raised if the dimension of the position doesn't match the dimension of
            the lattice.

        Parameters
        ----------
        shape: (N) array_like or int
            shape of finite size lattice to build.
        primitive: bool, optional
            If True the shape will be multiplied by the cell size of the model.
            The default is True.
        pos: (N) array_like or int, optional
            Optional position of the section to build. If ``None`` the origin is used.
        check: bool, optional
            If `True` the positions of the translation vectors are checked and
            filtered. The default is True. This should only be disabled if
            filtered later.
        dtype: int or np.dtype, optional
            Optional data-type for storing the lattice indices. By default, the given
            limits are checked to determine the smallest possible data-type.
        oversample: float, optional
            Faktor for upscaling limits for initial index grid. This ensures that all
            positions are included. Only needed if corner points are missing.
            The default is 0.

        Returns
        -------
        nvecs: (M, N) np.ndarray
            The translation-vectors in lattice-coordinates.

        Examples
        --------
        >>> latt = LatticeStructure(np.eye(2))
        >>> latt.build_translation_vectors((2, 2))
        [[0 0]
         [0 1]
         [0 2]
         [1 0]
         [1 1]
         [1 2]
         [2 0]
         [2 1]
         [2 2]]
        """
        # Build lattice indices
        if isinstance(shape, AbstractShape):
            pos, stop = shape.limits().T
            shape = stop - pos
        shape = np.atleast_1d(shape)
        if len(shape) != self.dim:
            raise ValueError(
                f"Dimension of shape {len(shape)} doesn't "
                f"match the dimension of the lattice {self.dim}"
            )
        logger.debug("Building nvecs: %s at %s", shape, pos)

        if primitive:
            shape = np.array(shape) * np.max(self.vectors, axis=0) - 0.1 * self.norms
        if pos is None:
            pos = np.zeros(self.dim)
        end = pos + shape

        # Estimate the maximum needed translation vector to reach all points
        max_nvecs = np.array(
            [self.itranslate(pos)[0], self.itranslate(end)[0]], dtype=np.float64
        )
        for i in range(1, self.dim):
            for idx in itertools.combinations(range(self.dim), r=i):
                _pos = end.copy()
                _pos[np.array(idx)] = 0
                index = self.itranslate(_pos)[0]
                max_nvecs[0] = np.min([index, max_nvecs[0]], axis=0)
                max_nvecs[1] = np.max([index, max_nvecs[1]], axis=0)
        # Pad maximum translation vectors and create index limits
        padding = oversample * shape + 1
        max_nvecs += [-padding, +padding]
        limits = max_nvecs.astype(np.int64).T
        logger.debug("Limits: %s, %s", limits[:, 0], limits[:, 1])

        # Generate translation vectors with too many points to reach each corner
        nvecs = vindices(limits, sort_axis=0, dtype=dtype)
        logger.debug("%s Translation vectors built", len(nvecs))
        if check:
            logger.debug("Filtering nvec's")
            # Filter points in the given volume
            positions = np.dot(nvecs, self.vectors[np.newaxis, :, :])[:, 0, :]
            mask = self.check_points(positions, shape, primitive, pos)
            nvecs = nvecs[mask]
        return nvecs

    # noinspection PyShadowingNames
    def build_indices(
        self,
        shape: Union[int, Sequence[int], AbstractShape],
        primitive: bool = False,
        pos: Union[float, Sequence[float]] = None,
        check: bool = True,
        callback: Callable = None,
        dtype: Union[int, str, np.dtype] = None,
        return_pos: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Constructs the lattice indices .math:`(n, α)` in the given shape.

        Raises
        ------
        ValueError
            Raised if the dimension of the position doesn't match the dimension
            of the lattice.

        Parameters
        ----------
        shape: (N) array_like or int or AbstractShape
            shape of finite size lattice to build.
        primitive: bool, optional
            If True the shape will be multiplied by the cell size of the model.
            The default is True.
        pos: (N) array_like or int, optional
            Optional position of the section to build. If ``None`` the origin is used.
        check : bool, optional
            If True the positions of the translation vectors are checked and
            filtered. The default is True. This should only be disabled if
            filtered later.
        callback: callable, optional
            Optional callable for filtering sites.
            The indices and positions are passed as arguments.
        dtype: int or str or np.dtype, optional
            Optional data-type for storing the lattice indices. By default, the given
            limits are checked to determine the smallest possible data-type.
        return_pos: bool, optional
            Flag if positions should be returned with the indices. This can speed up
            the building process, since the positions have to be computed here anyway.
            The default is ``False``.

        Returns
        -------
        indices: (M, N+1) np.ndarray
            The lattice indices of the sites in the format
            .math:`(n_1, ..., n_d, α)`.
        positions: (M, N) np.ndarray
            Corresponding positions. Only returned if ``return_positions=True``.

        Examples
        --------
        Build indices of a linear chain with two atoms in the unit cell:

        >>> latt = LatticeStructure(np.eye(2))
        >>> latt.add_atom([0.0, 0.0], "A")
        >>> latt.add_atom([0.5, 0.5], "B")
        >>> indices, positions = latt.build_indices((2, 1), return_pos=True)

        The indices contain the translation vector and the atom index

        >>> indices
        [[0 0 0]
         [0 0 1]
         [0 1 0]
         [1 0 0]
         [1 0 1]
         [1 1 0]
         [2 0 0]
         [2 1 0]]

        The positions are the positions of the atoms in the same order of the indices:

        >>> positions
        [[0.  0. ]
         [0.5 0.5]
         [0.  1. ]
         [1.  0. ]
         [1.5 0.5]
         [1.  1. ]
         [2.  0. ]
         [2.  1. ]]
        """
        logger.debug("Building lattice-indices: %s at %s", shape, pos)

        # Build lattice indices
        nvecs = self.build_translation_vectors(shape, primitive, pos, False, dtype)
        ones = np.ones(nvecs.shape[0], dtype=nvecs.dtype)
        arrays = [np.c_[nvecs, i * ones] for i in range(self.num_base)]
        cols = self.dim + 1
        indices = np.ravel(arrays, order="F")
        indices = indices.reshape(cols, int(indices.shape[0] / cols)).T

        logger.debug("Computing positions of sub-lattices")
        # Compute positions for filtering
        positions = [self.translate(nvecs, pos) for pos in self.atom_positions]
        positions = interweave(positions)

        if check:
            # Filter points in the given volume
            logger.debug("Filtering points")
            mask = self.check_points(positions, shape, primitive, pos)
            indices = indices[mask]
            positions = positions[mask]

        # Filter points with user method
        if callback is not None:
            logger.debug("Applying callback-method")
            mask = callback(indices, positions)
            indices = indices[mask]
            positions = positions[mask]

        logger.debug("Created %i lattice sites", len(indices))
        return indices, positions if return_pos else None

    def _filter_neighbors(self, indices, neighbors, distances, x_ind=None):
        logger.debug("Filtering neighbors")

        x_ind = indices if x_ind is None else x_ind

        # Add dummy index for invalid index
        invalid_ind = len(indices)
        indices_padded = np.append(indices, -np.ones((1, indices.shape[1])), axis=0)
        alphas = indices_padded[neighbors, -1]

        # Remove not connected neighbors
        dist_matrix = self._distance_matrix
        num_base = len(dist_matrix)
        for a1 in range(num_base):
            mask1 = (x_ind[:, -1] == a1)[:, None]
            for a2 in range(num_base):
                connected = np.isin(distances, dist_matrix[a1][a2])
                mask = mask1 & (alphas == a2) & (~connected)
                neighbors[mask] = invalid_ind
                distances[mask] = np.inf

        logger.debug("Re-sorting and cleaning up neighbors")

        # Resort neighbors
        i = np.arange(len(distances))[:, np.newaxis]
        j = np.argsort(distances, axis=1)
        distances = distances[i, j]
        neighbors = neighbors[i, j]
        num_sites = len(neighbors)

        if not np.any(distances == np.inf):
            # No invalid entries found. This usually happens for (2, 2, ..) systems
            # which results in a bug for the neighbor data:
            # Assume a 1D chain of 3 atoms. The outer two atoms each have only one
            # neighbor (the center atom), whereas the center atom has two.
            # In a model of two atoms no atom has two neighbors, which prevents
            # the algorithm to create an array of size (N, 2).
            # To prevent this a column of invalid values is appended to the data.
            # Since the system size is small this doesn't create any memory issues.
            shape = num_sites, 1
            neighbors = np.append(neighbors, np.full(shape, invalid_ind), axis=1)
            distances = np.append(distances, np.full(shape, np.inf), axis=1)
        else:
            # Remove columns containing only invalid data
            all_valid = np.any(distances != np.inf, axis=0)
            distances = distances[:, all_valid]
            neighbors = neighbors[:, all_valid]

        return neighbors, distances

    # noinspection PyShadowingNames
    def compute_neighbors(
        self, indices: ArrayLike, positions: ArrayLike, num_jobs: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the neighbors for the given points.

        Parameters
        ----------
        indices : (N, D+1) array_like
            The lattice indices of the sites in the format
            .math:`(n_1, ..., n_D, α)` where N is the number of sites and D the
            dimension of the lattice.
        positions : (N, D) array_like
            The positions of the sites in cartesian coordinates where N is the number
            of sites and D the dimension of the lattice.
        num_jobs : int, optional
            Number of jobs to schedule for parallel processing.
            If ``-1`` is given all processors are used. The default is ``1``.

        Returns
        -------
        neighbors: (N, M) np.ndarray
            The indices of the neighbors in ``positions``. M is the maximum
            number of neighbors previously computed in the ``analyze`` method.
        distances: (N, M) np.ndarray
            The corresponding distances of the neighbors.

        See Also
        --------
        analyze : Used to pre-compute the base neighbors of the unit cell.

        Examples
        --------
        Construct indices of a one dimensional lattice:

        >>> latt = LatticeStructure(1)
        >>> latt.add_atom()
        >>> latt.add_connections()
        >>> indices, positions = latt.build_indices(3, return_pos=True)
        >>> positions
        [[0.]
         [1.]
         [2.]
         [3.]]

        Compute the neighbors of the constructed sites

        >>> neighbors, distances = latt.compute_neighbors(indices, positions)
        >>> neighbors
        [[1 4]
         [2 0]
         [3 1]
         [2 4]]

        >>> indices
        [[ 1. inf]
         [ 1.  1.]
         [ 1.  1.]
         [ 1. inf]]

        The neighbor indices and distances of sites with less than the maximum number
        of neighbors are filled up with an invalid index (here: 4) and ``np.inf``
        as distance.
        """

        logger.debug("Querying neighbors of %i points", len(positions))

        # Set neighbor query parameters and build tree
        k = np.sum(np.sum(self._raw_num_neighbors, axis=1)) + 1
        max_dist = np.max(self.distances) + 0.1 * np.min(self._raw_distance_matrix)
        tree = KDTree(positions, k=k, max_dist=max_dist)
        logger.debug("Max. number of neighbors: %i", k)
        logger.debug("Max. neighbor distance:   %f", max_dist)
        # Query and filter neighbors
        neighbors, distances = tree.query(
            num_jobs=num_jobs, decimals=self.DIST_DECIMALS
        )
        neighbors, distances = self._filter_neighbors(indices, neighbors, distances)

        return neighbors, distances

    def copy(self) -> "LatticeStructure":
        """LatticeStructure : Creates a (deep) copy of the lattice instance."""
        return deepcopy(self)

    def todict(self) -> dict:
        """Creates a dictionary containing the information of the lattice instance.

        Returns
        -------
        d : dict
            The information defining the current instance.
        """
        d = dict()
        d["vectors"] = self.vectors
        d["atoms"] = self._atoms
        d["positions"] = self._positions
        d["connections"] = self._connections
        return d

    @classmethod
    def fromdict(cls, d):
        """Creates a new instance from information stored in a dictionary.

        Parameters
        ----------
        d : dict
            The information defining the current instance.

        Returns
        -------
        latt : LatticeStructure
            The restored lattice instance.
        """
        self = cls(d["vectors"])
        for pos, at in zip(d["positions"], d["atoms"]):
            self.add_atom(pos, at)
        self._connections = d["connections"]
        self.analyze()
        return self

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
        """Save the data of the ``LatticeStructure`` instance.

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
    def load(
        cls, file: Union[str, int, bytes]
    ) -> "LatticeStructure":  # pragma: no cover
        """Load data of a saved ``LatticeStructure`` instance.

        Parameters
        ----------
        file : str or int or bytes
            File name to load the lattice.

        Returns
        -------
        latt : LatticeStructure
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

    def plot_cell(
        self,
        lw: float = None,
        alpha: float = 0.5,
        cell_lw: float = None,
        cell_ls: str = "--",
        margins: Union[Sequence[float], float] = 0.1,
        legend: bool = None,
        grid: bool = False,
        show_cell: bool = True,
        show_vecs: bool = True,
        show_neighbors: bool = True,
        con_colors: Sequence = None,
        adjustable: str = "box",
        ax: Union[plt.Axes, Axes3D] = None,
        show: bool = False,
    ) -> Union[plt.Axes, Axes3D]:  # pragma: no cover
        """Plot the unit cell of the lattice.

        Parameters
        ----------
        lw : float, optional
            Line width of the neighbor connections.
        alpha : float, optional
            The alpha value of the neighbor sites.
        cell_lw : float, optional
            The line width used for plotting the unit cell outlines.
        cell_ls : str, optional
            The line style used for plotting the unit cell outlines.
        margins : Sequence[float] or float, optional
            The margins of the plot.
        legend : bool, optional
            Flag if legend is shown.
        grid : bool, optional
            If True, draw a grid in the plot.
        show_neighbors : bool, optional
            If True the neighbors are plotted.
        show_vecs : bool, optional
            If True the first unit-cell is drawn.
        show_cell : bool, optional
            If True the outlines of the unit cell are plotted.
        con_colors : Sequence[tuple], optional
            list of colors to override the defautl connection color. Each element
            has to be a tuple with the first two elements being the atom indices of
            the pair and the third element the color, for example ``[('A', 'A', 'r')]``.
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
        logger.debug("Plotting unit cell")
        if self.dim > 3:
            raise ValueError(f"Plotting in {self.dim} dimensions is not supported!")

        hopz, atomz = range(2)

        fig, ax = subplot(self.dim, adjustable, ax)

        # Draw unit vectors and the cell they spawn.
        if show_vecs:
            vectors = self.vectors
            draw_unit_cell(
                ax, vectors, show_cell, color="k", lw=cell_lw, ls=cell_ls, zorder=hopz
            )
        # Draw sites
        colors = list()
        for i in range(self.num_base):
            atom = self.atoms[i]
            col = atom.color or f"C{i}"
            points = self.atom_positions[i]
            rad = atom.radius
            draw_sites(ax, points, rad, color=col, label=atom.name, zorder=atomz)
            colors.append(col)
        # Draw Neighbors and connections
        ccolor = "k"
        hop_colors = connection_color_array(self.num_base, ccolor, con_colors)
        if show_neighbors:
            position_arr = [list() for _ in range(self.num_base)]
            for i in range(self.num_base):
                p1 = self.atom_positions[i]
                for distidx in range(self.num_distances):
                    try:
                        indices = self.get_neighbors(alpha=i, distidx=distidx)
                        positions = self.get_positions(indices)
                        for idx, p2 in zip(indices, positions):
                            j = idx[-1]
                            col = hop_colors[i][j]
                            draw_vectors(
                                ax, p2 - p1, pos=p1, zorder=hopz, color=col, lw=lw
                            )
                            if np.any(idx[:-1]):
                                position_arr[j].append(p2)
                    except IndexError:
                        pass
            for i in range(self.num_base):
                atom = self.get_atom(i)
                positions = position_arr[i]
                if positions:
                    pos = np.unique(positions, axis=0)
                    rad = 0.6 * atom.radius
                    col = colors[i]
                    draw_sites(ax, pos, rad, color=col, alpha=alpha, zorder=atomz)
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
        return (
            f"{self.__class__.__name__}("
            f"dim: {self.dim}, "
            f"num_base: {self.num_base}, "
            f"num_neighbors: {self.num_neighbors})"
        )
