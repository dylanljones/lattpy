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
import collections
import numpy as np
from copy import deepcopy
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from typing import Union, Optional, Tuple, List, Iterator, Sequence, Callable, Any, Dict

from .utils import (
    ArrayLike,
    frmt_num,
    SiteOccupiedError,
    NoAtomsError,
    NoBaseNeighborsError,
    NotBuiltError
)
from .spatial import (
    build_periodic_translation_vector,
    vindices,
    interweave,
    cell_size,
    cell_volume,
    WignerSeitzCell,
    KDTree
)
from .plotting import (
    draw_points,
    draw_vectors,
    draw_cell,
    draw_indices
)
from .unitcell import Atom
from .data import LatticeData, DataMap

__all__ = ["Lattice"]

logger = logging.getLogger(__name__)

vecs_t = Union[float, Sequence[float], Sequence[Sequence[float]]]


class Lattice:
    """Main lattice object representing a Bravais lattice.

    Parameters
    ----------
    vectors: array_like or float
        The primitive basis vectors that define the unit cell of the lattice.

    Examples
    --------
    Two dimensional lattice with one atom in the unit cell and nearest neighbors
    >>> latt = Lattice(np.eye(2))
    >>> latt.add_atom()
    >>> latt.add_connections(1)
    Lattice(dim: 2, num_base: 1, shape: None)
    """

    DIST_DECIMALS: int = 6        # Decimals used for rounding distances
    RVEC_TOLERANCE: float = 1e-6  # Tolerance for reciprocal vectors/lattice

    # noinspection PyUnusedLocal
    def __init__(self, vectors: vecs_t, **kwargs):
        # Vector basis
        self._vectors = np.atleast_2d(vectors).T
        self._vectors_inv = np.linalg.inv(self._vectors)
        self._dim = len(self._vectors)
        self._cell_size = cell_size(self.vectors)
        self._cell_volume = cell_volume(self.vectors)

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

        # Lattice Cache
        self.data = LatticeData()
        self.shape = None
        self.periodic_axes = list()
        logger.debug("Lattice initialized (D=%i)\nvectors:\n%s",
                     self.dim, self._vectors)

    @classmethod
    def chain(cls, a: float = 1.0, **kwargs) -> 'Lattice':
        return cls(a, **kwargs)

    @classmethod
    def square(cls, a: float = 1.0, **kwargs) -> 'Lattice':
        return cls(a * np.eye(2), **kwargs)

    @classmethod
    def rectangular(cls, a1: float = 1., a2: float = 1.,
                    **kwargs) -> 'Lattice':
        return cls(np.array([[a1, 0], [0, a2]]), **kwargs)

    @classmethod
    def hexagonal(cls, a: float = 1.0, **kwargs) -> 'Lattice':
        vectors = a / 2 * np.array([[3, np.sqrt(3)], [3, -np.sqrt(3)]])
        return cls(vectors, **kwargs)

    @classmethod
    def oblique(cls, alpha: float, a1: float = 1.0, a2: float = 1.0,
                **kwargs) -> 'Lattice':
        vectors = np.array([[a1, 0], [a2 * np.cos(alpha), a2 * np.sin(alpha)]])
        return cls(vectors, **kwargs)

    @classmethod
    def hexagonal3d(cls, a: float = 1., az: float = 1., **kwargs) -> 'Lattice':
        vectors = a / 2 * np.array([[3, np.sqrt(3), 0],
                                    [3, -np.sqrt(3), 0],
                                    [0, 0, az]])
        return cls(vectors, **kwargs)

    @classmethod
    def sc(cls, a: float = 1.0, **kwargs) -> 'Lattice':
        return cls(a * np.eye(3), **kwargs)

    @classmethod
    def fcc(cls, a: float = 1.0, **kwargs) -> 'Lattice':
        vectors = a / 2 * np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        return cls(vectors, **kwargs)

    @classmethod
    def bcc(cls, a: float = 1.0, **kwargs) -> 'Lattice':
        vectors = a / 2 * np.array([[1, 1, 1], [1, -1, 1], [-1, 1, 1]])
        return cls(vectors, **kwargs)

    # ==================================================================================

    @property
    def dim(self) -> int:
        """int : The dimension of the lattice."""
        return self._dim

    @property
    def vectors(self) -> np.ndarray:
        """np.ndarray : Array containing the basis vectors as rows."""
        return self._vectors.T

    @property
    def vectors3d(self) -> np.ndarray:
        """np.ndarray : The basis vectors expanded to three dimensions."""
        vectors = np.eye(3)
        vectors[:self.dim, :self.dim] = self._vectors
        return vectors.T

    @property
    def norms(self) -> np.ndarray:
        """np.ndarray : Lengths of the basis vectors."""
        return np.linalg.norm(self._vectors, axis=0)

    @property
    def cell_size(self) -> np.ndarray:
        """np.ndarray : The shape of the box spawned by the basis vectors."""
        return self._cell_size

    @property
    def cell_volume(self) -> float:
        """float : The volume of the unit cell defined by the basis vectors."""
        return self._cell_volume

    @property
    def num_base(self) -> int:
        """int : The number of atoms in the unit cell."""
        return self._num_base

    @property
    def atoms(self) -> List[Atom]:
        """list of Atom : List of the atoms in the unit cell."""
        return self._atoms

    @property
    def atom_positions(self) -> List[np.ndarray]:
        """list of np.ndarray : List of positions of the atoms in the unit cell."""
        return self._positions

    @property
    def num_distances(self) -> int:
        """int : The maximal number of distances between the lattice sites."""
        return int(np.max(self._connections))

    @property
    def num_neighbors(self) -> np.ndarray:
        """np.ndarray : The number of neighbors of each atom in the unitcell."""
        return self._num_neighbors

    @property
    def base_neighbors(self):
        """The neighbors of the unitcell at the origin."""
        return self._base_neighbors

    @property
    def distances(self) -> np.ndarray:
        """List of distances between the lattice sites."""
        return self._distances

    @property
    def num_sites(self) -> int:
        """int : Number of sites in lattice data (if lattice has been built)."""
        return self.data.num_sites

    @property
    def num_cells(self) -> int:
        """int : Number of unit-cells in lattice data (if lattice has been built)."""
        return np.unique(self.data.indices[:, :-1], axis=0).shape[0]

    def itransform(self, world_coords: Union[Sequence[int], Sequence[Sequence[int]]]
                   ) -> np.ndarray:
        """Transform the world coords (x, y, ...) into the basis coords (n, m, ...).

        Parameters
        ----------
        world_coords : (..., N) array_like
            The coordinates in the world coordinate system that are transformed
            into the lattice coordinate system.

        Returns
        -------
        basis_coords : (..., N) np.ndarray
            The coordinates in the lattice coordinate system.

        Examples
        --------
        Construct a lattice with basis vectors :math:`a_1 = (2, 0)` and
        :math:`a_2 = (0, 1)`:
        >>> latt = Lattice([[2, 0], [0, 1]])
        Transform points into the coordinat system of the lattice:
        >>> latt.itransform([2, 0])
        [1. 0.]
        >>> latt.itransform([4, 0])
        [2. 0.]
        >>> latt.itransform([0, 1])
        [0. 1.]
        """
        world_coords = np.atleast_1d(world_coords)
        return np.inner(world_coords, self._vectors_inv)

    def transform(self, basis_coords: Union[Sequence[int], Sequence[Sequence[int]]]
                  ) -> np.ndarray:
        """Transform the basis-coords (n, m, ...) into the world coords (x, y, ...).

        Parameters
        ----------
        basis_coords : (..., N) array_like
            The coordinates in the lattice coordinate system that are transformed
            into the world coordinate system.

        Returns
        -------
        world_coords : (..., N) np.ndarray
            The coordinates in the cartesian coordinate system.

        Examples
        --------
        Construct a lattice with basis vectors :math:`a_1 = (2, 0)` and
        :math:`a_2 = (0, 1)`:
        >>> latt = Lattice([[2, 0], [0, 1]])
        Transform points into the world coordinat system:
        >>> latt.itransform([1, 0])
        [2. 0.]
        >>> latt.itransform([2, 0])
        [4. 0.]
        >>> latt.itransform([0, 1])
        [0. 1.]
        """
        basis_coords = np.atleast_1d(basis_coords)
        return np.inner(basis_coords, self._vectors)

    def translate(self, nvec: Union[int, Sequence[int], Sequence[Sequence[int]]],
                  r: Union[float, Sequence[float]] = 0.0) -> np.ndarray:
        r"""Translates the given postion vector `r` by the translation vector `n`.

        The position is calculated using the translation vector :math:`n` and the
        atom position in the unitcell _math:`r`:
        .. math::
            R = \sum_i n_i v_i + r

        Parameters
        ----------
        nvec : (..., N) array_like
            Translation vector in the lattice coordinate system.
        r : (N) array_like, optional
            The position in cartesian coordinates. If no vector is passed only
            the translation is returned.

        Returns
        -------
        r_trans : (N) np.ndarray
            The translated position.

        Examples
        --------
        Construct a lattice with basis vectors :math:`a_1 = (2, 0)` and
        :math:`a_2 = (0, 1)`:
        >>> latt = Lattice([[2, 0], [0, 1]])
        Translate the origin:
        >>> n = [1, 0]
        >>> latt.translate(n)
        [2. 0.]

        Translate a point:
        >>> p = [0.5, 0.5]
        >>> latt.translate(n, p)
        [2.5 0.5]

        Translate a point by multiple translation vectors:
        >>> p = [0.5, 0.5]
        >>> nvecs = [[0, 0], [1, 0], [2, 0]]
        >>> latt.translate(nvecs, p)
        [[0.5 0.5]
         [2.5 0.5]
         [4.5 0.5]]
        """
        r = np.atleast_1d(r)
        nvec = np.atleast_1d(nvec)
        return r + np.inner(nvec, self._vectors)

    def itranslate(self, x: Union[float, Sequence[float]]) -> [np.ndarray, np.ndarray]:
        """Returns the translation vector and atom position of the given position.

        Parameters
        ----------
        x : (N) array_like or float
            Position vector in cartesian coordinates.

        Returns
        -------
        nvec : (N) np.ndarray
            Translation vector in the lattice basis.
        r : (N) np.ndarray, optional
            The position in real-space.

        Examples
        --------
        Construct a lattice with basis vectors :math:`a_1 = (2, 0)` and
        :math:`a_2 = (0, 1)`:
        >>> latt = Lattice([[2, 0], [0, 1]])
        >>> latt.itranslate([2, 0])
        (array([1., 0.]), array([0., 0.]))
        >>> latt.itranslate([2.5, 0.5])
        (array([1., 0.]), array([0.5, 0.5]))
        """
        x = np.atleast_1d(x)
        itrans = self.itransform(x)
        nvec = np.floor(itrans)
        r = x - self.translate(nvec)
        return nvec, r

    def is_reciprocal(self, vecs: vecs_t, tol: float = RVEC_TOLERANCE) -> bool:
        r"""Checks if the given vectors are reciprocal to the lattice vectors.

        The lattice- and reciprocal vectors :math:'a_i' and :math:'b_i' must satisfy
        the relation
        .. math::
            a_i \cdot b_i = 2 \pi \delta_{ij}

        To check the given vectors, the difference of each dot-product is compared to
        .math:'2\pi' with the given tolerance.

        Parameters
        ----------
        vecs : array_like or float
            The vectors to check. Must have the same dimension as the lattice.
        tol : float, optional
            The tolerance used for checking the result of the dot-products.

        Returns
        -------
        is_reciprocal : bool
            Flag if the vectors are reciprocal to the lattice basis vectors.
        """
        vecs = np.atleast_2d(vecs)
        two_pi = 2 * np.pi
        for a, b in zip(self.vectors, vecs):
            if abs(np.dot(a, b) - two_pi) > tol:
                return False
        return True

    def reciprocal_vectors(self, tol: float = RVEC_TOLERANCE,
                           check: bool = False) -> np.ndarray:
        r"""Computes the reciprocal basis vectors of the bravais lattice.

        The lattice- and reciprocal vectors :math:'a_i' and :math:'b_i' must satisfy
        the relation
        .. math::
            a_i \cdot b_i = 2 \pi \delta_{ij}

        Parameters
        ----------
        tol : float, optional
            The tolerance used for checking the result of the dot-products.
        check : bool, optional
            Check the result and raise an exception if it does not satisfy
            the definition.
        Returns
        -------
        v_rec : np.ndarray
            The reciprocal basis vectors of the lattice.

        Examples
        --------
        Reciprocal vectors of the square lattice:
        >>> latt = Lattice(np.eye(2))
        >>> latt.reciprocal_vectors()
        [[6.28318531 0.        ]
         [0.         6.28318531]]
        """
        two_pi = 2 * np.pi
        if self.dim == 1:
            return np.array([[two_pi / self.vectors[0, 0]]])

        # Convert basis vectors of the bravais lattice to 3D, compute
        # reciprocal vectors and convert back to actual dimension.
        a1, a2, a3 = self.vectors3d
        factor = 2 * np.pi / self.cell_volume
        b1 = np.cross(a2, a3)
        b2 = np.cross(a3, a1)
        b3 = np.cross(a1, a2)
        rvecs = factor * np.asarray([b1, b2, b3])
        rvecs = rvecs[:self.dim, :self.dim]

        # Fix the sign so that the dot-products are all positive
        # and raise an exception if anything went wrong
        vecs = self.vectors
        for i in range(self.dim):
            dot = np.dot(vecs[i], rvecs[i])
            # Check if dot product is - 2 pi
            if abs(dot + two_pi) <= tol:
                rvecs[i] *= -1
            # Raise an exception if checks are enabled and
            # dot product results in anything other than +2 pi
            elif check and abs(dot - two_pi) > tol:
                raise ValueError(f"{rvecs[i]} not a reciprocal vector to {vecs[i]}")

        return rvecs

    # noinspection PyShadowingNames
    def reciprocal_lattice(self, min_negative: bool = False) -> 'Lattice':
        """Creates the lattice in reciprocal space.

        Parameters
        ----------
        min_negative : bool, optional
            If 'True' the reciprocal vectors are scaled such that
            there are fewer negative elements than positive ones.

        Returns
        -------
        rlatt : Lattice
            The lattice in reciprocal space

        See Also
        --------
        reciprocal_vectors : Constructs the reciprocal vectors used for the
            reciprocal lattice

        Examples
        --------
        Reciprocal lattice of the square lattice:
        >>> latt = Lattice(np.eye(2))
        >>> rlatt = latt.reciprocal_lattice()
        >>> rlatt.vectors
        [[6.28318531 0.        ]
         [0.         6.28318531]]
        """
        rvecs = self.reciprocal_vectors(min_negative)
        rlatt = self.__class__(rvecs)
        return rlatt

    def get_neighbor_cells(self, distidx: int = 0,
                           include_origin: bool = True,
                           comparison: Callable = np.isclose) -> np.ndarray:
        """Find all neighboring unit cells of the unit cell at the origin.

        Parameters
        ----------
        distidx : int, default
            Index of distance to neighboring cells, default is 0 (nearest neighbors).
        include_origin : bool, optional
            If `True` the origin is included in the set.
        comparison : callable, optional
            The method used for comparing distances.

        Returns
        -------
        indices : np.ndarray
            The lattice indeices of the neighboring unit cells.

        Examples
        --------
        >>> latt = Lattice(np.eye(2))
        >>> latt.get_neighbor_cells(distidx=0, include_origin=False)
        [[-1  0]
         [ 0 -1]
         [ 0  1]
         [ 1  0]]
        """
        # Build cell points
        max_factor = distidx + 1
        axis_factors = np.arange(-max_factor, max_factor + 1)
        factors = np.array(list(itertools.product(axis_factors, repeat=self.dim)))
        points = np.dot(factors, self.vectors[np.newaxis, :, :])[:, 0, :]

        # Compute distances to origin for all points
        distances = np.linalg.norm(points, axis=1)

        # Set maximum distances value to number of neighbors
        # + number of unique vector lengths
        max_distidx = distidx + len(np.unique(np.linalg.norm(self.vectors, axis=1)))

        # Return points with distance lower than maximum distance
        maxdist = np.sort(np.unique(distances))[max_distidx]
        indices = np.where(comparison(distances, maxdist))[0]
        factors = factors[indices]

        origin = np.zeros(self.dim)
        idx = np.where((factors == origin).all(axis=1))[0]
        if include_origin and not len(idx):
            factors = np.append(origin[np.newaxis, :], factors, axis=0)
        elif not include_origin and len(idx):
            factors = np.delete(factors, idx, axis=0)
        return factors

    def wigner_seitz_cell(self) -> WignerSeitzCell:
        """Computes the Wigner-Seitz cell of the lattice structure.

        Returns
        -------
        ws_cell : WignerSeitzCell
            The Wigner-Seitz cell of the lattice.
        """
        nvecs = self.get_neighbor_cells(include_origin=True)
        positions = np.dot(nvecs, self.vectors[np.newaxis, :, :])[:, 0, :]
        return WignerSeitzCell(positions)

    def brillouin_zone(self, min_negative: bool = False) -> WignerSeitzCell:
        """Computes the first Brillouin-zone of the lattice structure.

        Constructs the Wigner-Seitz cell of the reciprocal lattice

        Parameters
        ----------
        min_negative : bool, optional
            If 'True' the reciprocal vectors are scaled such that
            there are fewer negative elements than positive ones.

        Returns
        -------
        ws_cell : WignerSeitzCell
            The Wigner-Seitz cell of the reciprocal lattice.
        """
        rvecs = self.reciprocal_vectors(min_negative)
        rlatt = self.__class__(rvecs)
        return rlatt.wigner_seitz_cell()

    # ==================================================================================

    def add_atom(self, pos: Union[float, Sequence[float]] = None,
                 atom: Union[str, Dict[str, Any], Atom] = None,
                 relative: bool = False,
                 neighbors: int = 0,
                 **kwargs) -> Atom:
        """Adds a site to the basis of the lattice unit cell.

        Parameters
        ----------
        pos : (N) array_like or float, optional
            Position of site in the unit-cell. The default is the origin of the cell.
            The size of the array has to match the dimension of the lattice.
        atom : str or dict or Atom, optional
            Identifier of the site. If a string is passed, a new Atom instance is
            created.
        relative : bool, optional
            Flag if the specified position is in cartesian or lattice coordinates.
            If `True` the passed position will be multiplied with the lattice vectors.
            The default is `False` (cartesian coordinates).
        neighbors : int, optional
            The number of neighbor distance to calculate. If the number is ´0´ the
            distances have to be calculated manually after configuring the
            lattice basis.
        **kwargs
            Keyword arguments for ´Atom´ constructor. Only used if a new `Atom`
            instance is created.

        Returns
        -------
        atom : Atom
            The `Atom`-instance of the newly added site.

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
        >>> latt = Lattice(np.eye(2))

        Create an atom and add it to the origin of the unit cell of the lattice
        >>> atom1 = Atom(name="A")
        >>> latt.add_atom([0.0, 0.0], atom=atom1)
        >>> latt.get_atom(0)
        Atom(A, size=10, 0)

        An `Atom` instance can also be created by passing the name of the (new) atom
        and optional keyword arguments for the constructor:
        >>> latt.add_atom([0.5, 0.5], atom="B", size=15)
        >>> latt.get_atom(1)
        Atom(B, size=15, 1)
        """
        pos = np.zeros(self.dim) if pos is None else np.atleast_1d(pos)
        if relative:
            pos = self.translate(pos)

        if len(pos) != self._dim:
            raise ValueError(f"Shape of the position {pos} doesn't match "
                             f"the dimension {self.dim} of the lattice!")
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
            The argument for getting the atom. If a `int` is passed
            it is interpreted as the index, if a `str` is passed as
            the name of an atom.

        Returns
        -------
        alpha : int or list of int
            The atom indices. If a string was passed multiple atoms with the same name
            can be returned as list.

        Examples
        --------
        Construct a lattice with two identical atoms and a third atom in the unit cell:
        >>> latt = Lattice(np.eye(2))
        >>> latt.add_atom([0, 0], atom="A")
        >>> latt.add_atom([0.5, 0], atom="B")
        >>> latt.add_atom([0.5, 0.5], atom="B")

        Get the atom index of atom `A`:
        >>> latt.get_alpha("A")
        [0]

        Get the indices of the atoms `B`:
        >>> latt.get_alpha("B")
        [1, 2]

        Since there are two atoms `B` in the unit cell of the lattice both indices
        are returned.
        """
        if isinstance(atom, Atom):
            return self._atoms.index(atom)
        elif isinstance(atom, str):
            return [i for i, at in enumerate(self._atoms) if atom == at.name]
        return atom

    def get_atom(self, atom: Union[int, str, Atom]) -> Atom:
        """Returns the `Atom` instance of the given atom in the unit cell.

        Parameters
        ----------
        atom : int or str or Atom
            The argument for getting the atom. If a `int` is passed
            it is interpreted as the index, if a `str` is passed as
            the name of an atom.

        Returns
        -------
        atom : Atom
            The `Atom` instance of the given site.

        See Also
        --------
        get_alpha : Get the index of the given atom.

        Examples
        --------
        Construct a lattice with one atom in the unit cell
        >>> latt = Lattice(np.eye(2))
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

    def add_connection(self, atom1: Union[int, str, Atom], atom2: Union[int, str, Atom],
                       num_distances=1, analyze: bool = False) -> None:
        """Sets the number of distances for a specific connection between two atoms.

        Parameters
        ----------
        atom1 : int or str or Atom
            The first atom of the connected pair.
        atom2 : int or str or Atom
            The second atom of the connected pair.
        num_distances : int, optional
            The number of neighbor-distance levels, e.g. setting to `1` means
            only nearest neighbors. The default are nearest neighbor connections.
        analyze : bool, optional
            If `True` the lattice basis is analyzed after adding connections.
            If `False` the `analyze`-method needs to be called manually.
            The default is `False`.

        See Also
        --------
        add_connections : Set up connections for all atoms in the unit cell in one call.
        analyze : Called after setting up all the lattice connections.

        Examples
        --------
        Construct a square lattice with two atoms, A and B, in the unit cell:
        >>> latt = Lattice(np.eye(2))
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
            The number of neighbor-distance levels, e.g. setting to `1` means
            only nearest neighbors. The default are nearest neighbor connections.
        analyze : bool, optional
            If `True` the lattice basis is analyzed after adding connections.
            If `False` the `analyze`-method needs to be called manually.
            The default is `True`.

        See Also
        --------
        add_connection : Set up connection between two specific atoms in the unit cell.
        analyze : Called after setting up all the lattice connections.

        Examples
        --------
        Construct a square lattice with one atom in the unit cell:
        >>> latt = Lattice(np.eye(2))
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
            e.g. setting to `1` means only nearest neighbors.
        analyze: bool
            Flag if lattice base is analyzed. If `False` the `analyze`-method
            needs to be called manually. The default is `True`.
        """
        warnings.warn("Configuring neighbors with `set_num_neighbors` is deprecated "
                      "and will be removed in a future version. Use the "
                      "`add_connections` instead.",
                      DeprecationWarning)
        self.add_connections(num_neighbors, analyze)

    def _compute_base_neighbors(self, max_distidx, num_jobs=1):
        logger.debug("Building indices of neighbor-cells")

        # Build indices of neighbor-cells
        self._positions = np.asarray(self._positions)
        cell_range = 2 * max_distidx
        logger.debug("Max. distidx: %i, Cell-range: %i", max_distidx, cell_range)

        nvecs = self.get_neighbor_cells(cell_range, include_origin=True,
                                        comparison=np.less_equal)
        arrays = [np.c_[nvecs, i * np.ones(nvecs.shape[0])]
                  for i in range(self.num_base)]
        cols = self.dim + 1
        indices = np.ravel(arrays, order="F").astype(np.int64)
        indices = indices.reshape(cols, int(indices.shape[0] / cols)).T

        # Compute positions and initialize tree
        nvecs, alphas = indices[:, :-1], indices[:, -1]
        positions = self.translate(nvecs, np.array(self.atom_positions)[alphas])
        tree = KDTree(positions, k=len(positions))

        # Compute neighbors for each distance level in the cell range
        n = self.num_base
        neighbor_array = [[collections.OrderedDict()
                           for _ in range(n)] for _ in range(n)]
        for a1 in range(self.num_base):
            pos = self.atom_positions[a1]
            neighbors, distances = tree.query(pos, num_jobs, self.DIST_DECIMALS,
                                              include_zero=True, compact=False)
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

    def analyze(self) -> None:
        """Analyzes the structure of the lattice and stores neighbor data.

        Check's distances between all sites of the bravais lattice and saves the `n`
        lowest values. The neighbor lattice-indices of the unit-cell are also stored
        for later use. This speeds up many calculations like finding nearest neighbors.

        Raises
        ------
        NoAtomsError
            Raised if no atoms where added to the lattice. The atoms in the unit cell
            are needed for computing the neighbors and distances of the lattice.

        Notes
        -----
        Before calling the `analyze` function all connections in the lattice have to
        be set up.

        Examples
        --------
        Construct a square lattice with one atom in the unit cell and nearest neighbors:
        >>> latt = Lattice(np.eye(2))
        >>> latt.add_atom()
        >>> latt.add_connections(num_distances=1)

        Call `analyze` after setting up the connections
        >>> latt.analyze()
        """
        logger.debug("Analyzing lattice")

        if len(self._atoms) == 0:
            raise NoAtomsError()

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
            distances[alpha, :len(dists)] = sorted(dists)

        self._base_neighbors = base_neighbors
        self._distance_matrix = base_distance_matrix
        self._num_neighbors = num_neighbors
        self._distances = distances
        logger.debug("Number of neighbors:\n%s", num_neighbors)
        logger.debug("Distance-matrix:\n%s", base_distance_matrix)
        logger.debug("Distances:\n%s", distances)

    def get_position(self, nvec: Union[int, Sequence[int]] = None,
                     alpha: int = 0) -> np.ndarray:
        """Returns the position for a given translation vector `nvec` and atom `alpha`.

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
        >>> latt = Lattice(np.eye(2))
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
            List of lattice indices in the format :math:'(n_1, ..., n_d, \alpha)'.

        Returns
        -------
        pos : (N, D) np.ndarray
            The positions of the lattice sites.

        Examples
        --------
        >>> latt = Lattice(np.eye(2))
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
        >>> latt = Lattice(np.eye(2))
        >>> latt.add_atom()
        >>> latt.add_connections(1)
        >>> latt.estimate_index([1.2, 0.2])
        [1 0]
        """
        pos = np.asarray(pos)
        n = np.asarray(np.round(self._vectors_inv @ pos, decimals=0), dtype="int")
        return n

    def get_neighbors(self, nvec: Union[int, Sequence[int]] = None,
                      alpha: int = 0,
                      distidx: int = 0) -> np.ndarray:
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
        NoBaseNeighboursError
            Raised if the lattice distances and base-neighbors haven't been computed.

        Examples
        --------
        >>> latt = Lattice(np.eye(2))
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
        if not self._base_neighbors:
            raise NoBaseNeighborsError()
        logger.debug("Computing neighbor-indices of %s, %i (distidx: %i)",
                     nvec, alpha, distidx)

        nvec = np.atleast_1d(nvec)
        keys = list(sorted(self._base_neighbors[alpha].keys()))
        dist = keys[distidx]
        indices = self._base_neighbors[alpha][dist]
        indices_transformed = indices.copy()
        indices_transformed[:, :-1] += nvec.astype(np.int64)
        logger.debug("Neighbour-indices: %s", indices_transformed)

        return indices_transformed

    def get_neighbor_positions(self, nvec: Union[int, Sequence[int]] = None,
                               alpha: int = 0,
                               distidx: int = 0) -> np.ndarray:
        """Returns the neighour positions of a given site by transforming neighbor data.

        Raises
        ------
        NoBaseNeighboursError
            Raised if the lattice distances and base-neighbors haven't been computed.

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
        NoBaseNeighboursError
            Raised if the lattice distances and base-neighbors haven't been computed.

        Examples
        --------
        >>> latt = Lattice(np.eye(2))
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
        if not self._base_neighbors:
            raise NoBaseNeighborsError()
        logger.debug("Computing neighbor-positions of %s, %i (distidx: %i)",
                     nvec, alpha, distidx)

        indices = self.get_neighbors(nvec, alpha, distidx)
        nvecs, alphas = indices[:, :-1], indices[:, -1]
        atom_pos = self._positions[alphas]
        positions = self.translate(nvecs, atom_pos)
        logger.debug("Neighbour-positions: %s", positions)

        return positions

    def get_neighbor_vectors(self, alpha: int = 0, distidx: int = 0,
                             include_zero: bool = False) -> np.ndarray:
        """Returns the vectors to the neigbor sites of an atom in the unit cell.

        Parameters
        ----------
        alpha : int, optional
            Index of the base atom. The default is the first atom in the unit cell.
        distidx : int, default
            Index of distance to the neighbors, default is 0 (nearest neighbors).
        include_zero : bool, optional
            Flag if zero-vector is included in result. The default is False.

        Returns
        -------
        vectors : np.ndarray
            The vectors from the site of the atom :math:`alpha` to the neighbor sites.

        Raises
        ------
        NoBaseNeighboursError
            Raised if the lattice distances and base-neighbors haven't been computed.

        Examples
        --------
        >>> latt = Lattice(np.eye(2))
        >>> latt.add_atom()
        >>> latt.add_connections(1)
        >>> latt.get_neighbor_vectors(alpha=0, distidx=0)
        [[ 1.  0.]
         [ 0. -1.]
         [-1.  0.]
         [ 0.  1.]]
        """
        if not self._base_neighbors:
            raise NoBaseNeighborsError()
        logger.debug("Computing neighbor-vectors of atom %i (distidx: %i)",
                     alpha, distidx)

        pos0 = self._positions[alpha]
        pos1 = self.get_neighbor_positions(alpha=alpha, distidx=distidx)
        if include_zero:
            pos1 = np.append(np.zeros((1, self.dim)), pos1, axis=0)
        vecs = pos1 - pos0
        logger.debug("Neighbour-vectors: %s", vecs)

        return vecs

    def fourier_weights(self, k: ArrayLike, alpha: int = 0,
                        distidx: int = 0) -> np.ndarray:
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

    def get_base_atom_dict(self, atleast2d: bool = True) -> \
            Dict[Any, List[Union[np.ndarray, Any]]]:
        """Returns a dictionary containing the positions for eatch of the base atoms.

        Parameters
        ----------
        atleast2d : bool, optional
            If 'True', one-dimensional coordinates will be cast to 2D vectors.

        Returns
        -------
        atom_pos : dict
            The positions of the atoms as a dictionary.

        Examples
        --------
        >>> latt = Lattice(np.eye(2))
        >>> latt.add_atom([0, 0], atom="A")
        >>> latt.add_atom([0.5, 0], atom="B")
        >>> latt.add_atom([0.5, 0.5], atom="B")
        >>> latt.get_base_atom_dict()
        {
            Atom(A, size=10, 0): [array([0, 0])],
            Atom(B, size=10, 1): [array([0.5, 0. ]), array([0.5, 0.5])]
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

    def build_translation_vectors(self, shape: Union[int, Sequence[int]],
                                  relative: bool = False,
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
        relative: bool, optional
            If 'True' the shape will be multiplied by the cell size of the model.
            The default is `True`.
        pos: (N) array_like or int, optional
            Optional position of the section to build. If 'None' the origin is used.
        check: bool, optional
            If `True` the positions of the translation vectors are checked and
            filtered. The default is `True`. This should only be disabled if
            filtered later.
        dtype: int or np.dtype, optional
            Optional data-type for storing the lattice indices. By default, the given
            limits are checked to determine the smallest possible data-type.
        oversample: float, optional
            Faktor for upscaling limits for initial index grid. This ensures that all
            positions are included. Only needed if corner points are missing.
            The default is `0`.

        Returns
        -------
        nvecs: (M, N) np.ndarray
            The translation-vectors in lattice-coordinates.

        Examples
        --------
        >>> latt = Lattice(np.eye(2))
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
        shape = np.atleast_1d(shape)
        if len(shape) != self.dim:
            raise ValueError(f"Dimension of shape {len(shape)} doesn't "
                             f"match the dimension of the lattice {self.dim}")
        logger.debug("Building nvecs: %s at %s", shape, pos)

        if relative:
            shape = np.array(shape) * np.max(self.vectors, axis=0) - 0.1 * self.norms
        if pos is None:
            pos = np.zeros(self.dim)
        end = pos + shape

        # Estimate the maximum needed translation vector to reach all points
        max_nvecs = np.array([self.itranslate(pos)[0], self.itranslate(end)[0]])
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
            mask = (pos[0] <= positions[:, 0]) & (positions[:, 0] <= end[0])
            for i in range(1, self.dim):
                mask = mask & (pos[i] <= positions[:, i]) & (positions[:, i] <= end[i])
            nvecs = nvecs[mask]
        return nvecs

    # noinspection PyShadowingNames
    def check_points(self, points: np.ndarray,
                     shape: Union[int, Sequence[int]],
                     relative: bool = False,
                     pos: Union[float, Sequence[float]] = None,
                     eps: float = 1e-3,
                     ) -> np.ndarray:
        """Returns a mask for the points in the given shape.

        Parameters
        ----------
        points: (M, N) np.ndarray
            The points in cartesian coordinates.
        shape: (N) array_like or int
            shape of finite size lattice to build.
        relative: bool, optional
            If 'True' the shape will be multiplied by the cell size of the model.
            The default is `True`.
        pos: (N) array_like or int, optional
            Optional position of the section to build. If 'None' the origin is used.
        eps: float, optional
            Optional padding of the shape for checking the points.
            The default is `1e-3`.

        Returns
        -------
        mask: (M) np.ndarray
            The mask for the points inside the shape.

        Examples
        --------
        >>> latt = Lattice(np.eye(2))
        >>> shape = (2, 2)
        >>> points = np.array([[0, 0], [2, 2], [3, 2]])
        >>> latt.check_points(points, shape)
        [ True  True False]
        """
        shape = np.atleast_1d(shape)
        if len(shape) != self.dim:
            raise ValueError(f"Dimension of shape {len(shape)} doesn't "
                             f"match the dimension of the lattice {self.dim}")
        if relative:
            shape = np.array(shape) * np.max(self.vectors, axis=0) - 0.1 * self.norms

        pos = np.zeros(self.dim) if pos is None else np.array(pos, dtype=np.float64)
        pos -= eps
        end = pos + shape + eps

        mask = (pos[0] <= points[:, 0]) & (points[:, 0] <= end[0])
        for i in range(1, self.dim):
            mask = mask & (pos[i] <= points[:, i]) & (points[:, i] <= end[i])
        return mask

    # noinspection PyShadowingNames
    def build_indices(self, shape: Union[int, Sequence[int]],
                      relative: bool = False,
                      pos: Union[float, Sequence[float]] = None,
                      check: bool = True,
                      callback: Callable = None,
                      dtype: Union[int, str, np.dtype] = None,
                      return_pos: bool = False
                      ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Constructs the lattice indices .math:`(n, \alpha)` in the given shape.

        Raises
        ------
        ValueError
            Raised if the dimension of the position doesn't match the dimension
            of the lattice.

        Parameters
        ----------
        shape: (N) array_like or int
            shape of finite size lattice to build.
        relative: bool, optional
            If 'True' the shape will be multiplied by the cell size of the model.
            The default is `True`.
        pos: (N) array_like or int, optional
            Optional position of the section to build. If 'None' the origin is used.
        check : bool, optional
            If `True` the positions of the translation vectors are checked and filtered.
            The default is `True`. This should only be disabled if filtered later.
        callback: callable, optional
            Optional callable for filtering sites.
            The indices and positions are passed as arguments.
        dtype: int or str or np.dtype, optional
            Optional data-type for storing the lattice indices. By default the given
            limits are checked to determine the smallest possible data-type.
        return_pos: bool, optional
            Flag if positions should be returned with the indices. This can speed up
            the building process, since the positions have to be computed here anyway.
            The default is `False`.

        Returns
        -------
        indices: (M, N+1) np.ndarray
            The lattice indices of the sites in the format
            .math:`(n_1, ..., n_d, \alpha)`.
        positions: (M, N) np.ndarray
            Corresponding positions. Only returned if `return_positions` equals `True`.

        Examples
        --------
        Build indices of a linear chain with two atoms in the unit cell:
        >>> latt = Lattice(np.eye(2))
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
        nvecs = self.build_translation_vectors(shape, relative, pos, check, dtype)
        ones = np.ones(nvecs.shape[0], dtype=nvecs.dtype)
        arrays = [np.c_[nvecs, i * ones] for i in range(self.num_base)]
        cols = self.dim + 1
        indices = np.ravel(arrays, order="F")
        indices = indices.reshape(cols, int(indices.shape[0] / cols)).T

        logger.debug("Computing positions of sub-lattices")
        # Compute positions for filtering
        positions = [self.translate(nvecs, pos) for pos in self.atom_positions]
        positions = interweave(positions)

        # Filter points in the given volume
        logger.debug("Filtering points")
        mask = self.check_points(positions, shape, relative, pos)
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
    def compute_neighbors(self, indices: ArrayLike, positions: ArrayLike,
                          num_jobs: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the neighbors for the given points.

        Parameters
        ----------
        indices : (N, D+1) array_like
            The lattice indices of the sites in the format
            .math:`(n_1, ..., n_D, \alpha)` where N is the number of sites and D the
            dimension of the lattice.
        positions : (N, D) array_like
            The positions of the sites in cartesian coordinates where N is the number
            of sites and D the dimension of the lattice.
        num_jobs : int, optional
            Number of jobs to schedule for parallel processing.
            If -1 is given all processors are used. The default is `1`.

        Returns
        -------
        neighbors: (N, M) np.ndarray
            The indices of the neighbors in `positions`. M is the maximum
            number of neighbors previously computed in the `analyze` method.
        distances: (N, M) np.ndarray
            The corresponding distances of the neighbors.

        See Also
        --------
        analyze : Used to pre-compute the base neighbors of the unit cell.

        Examples
        --------
        Construct indices of a one dimensional lattice:
        >>> latt = Lattice(1)
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
        of neighbors are filled up with an invalid index (here: 4) and `np.inf`
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
        neighbors, distances = tree.query(num_jobs=num_jobs,
                                          decimals=self.DIST_DECIMALS)
        neighbors, distances = self._filter_neighbors(indices, neighbors, distances)

        return neighbors, distances

    # ==================================================================================
    # Cached lattice
    # ==================================================================================

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
            The lattice index (n_1, ..., n_D, alpha) of the site.

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

    def neighbors(self, site: int, distidx: int = None,
                  unique: bool = False) -> np.ndarray:
        """Returns the neighours of a given site in the cached lattice data.

        Parameters
        ----------
        site : int
            The super-index of a site in the cached lattice data.
        distidx : int, optional
            Index of distance to the neighbors, default is 0 (nearest neighbors).
        unique : bool, optional
            If 'True', each unique pair is only returned once.

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
            If 'True', each unique pair is only return once.

        Returns
        -------
        indices : (N, ) np.ndarray of int
            The super-indices of the nearest neighbors.
        """
        return self.neighbors(idx, 0, unique)

    def iter_neighbors(self, site: int, unique: bool = False
                       ) -> Iterator[Tuple[int, np.ndarray]]:
        """Iterates over the neighbors of all distances of a given site.

        Parameters
        ----------
        site : int
            The super-index of a site in the cached lattice data.
        unique : bool, optional
            If 'True', each unique pair is only return once.


        Yields
        -------
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

    def build(self, shape: Union[int, Sequence[int]],
              relative: bool = False,
              pos: Union[float, Sequence[float]] = None,
              check: bool = True,
              num_jobs: int = -1,
              periodic: Union[bool, int, Sequence[int]] = None,
              callback: Callable = None,
              dtype: Union[int, str, np.dtype] = None
              ) -> LatticeData:
        """Constructs the indices and neighbors of a finite size lattice.

        Parameters
        ----------
        shape : (N, ) array_like or int
            shape of finite size lattice to build.
        relative : bool, optional
            If 'True' the shape will be multiplied by the cell size of the model.
            The default is `True`.
        pos : (N, ) array_like or int, optional
            Optional position of the section to build. If 'None' the origin is used.
        check : bool, optional
            If `True` the positions of the translation vectors are checked and filtered.
            The default is `True`. This should only be disabled if filtered later.
        num_jobs : int, optional
            Number of jobs to schedule for parallel processing of neighbors.
            If -1 is given all processors are used. The default is `-1`.
        periodic : int or array_like, optional
            Optional periodic axes to set. See 'set_periodic' for mor details.
        callback : callable, optional
            The indices and positions are passed as arguments.
        dtype : int or str or np.dtype, optional
            Optional data-type for storing the lattice indices. Using a smaller
            bit-size may help reduce memory usage. By default the given limits are
            checked to determine the smallest possible data-type.

        Raises
        ------
        ValueError
            Raised if the dimension of the position doesn't match the dimension of
            the lattice.
        NoBaseNeighboursError
            Raised if the lattice distances and base-neighbors haven't been computed.
        """
        self.data.reset()
        shape = np.atleast_1d(shape)
        if len(shape) != self.dim:
            raise ValueError(f"Dimension of shape {len(shape)} doesn't "
                             f"match the dimension of the lattice {self.dim}")
        if not self._base_neighbors:
            raise NoBaseNeighborsError()
        logger.debug("Building lattice: %s at %s", shape, pos)

        # Build indices and positions
        indices, positions = self.build_indices(shape, relative, pos, check,
                                                callback, dtype, True)

        # Compute the neighbors and distances between the sites
        neighbors, distances = self.compute_neighbors(indices, positions, num_jobs)

        # Set data of the lattice and update shape
        self.data.set(indices, positions, neighbors, distances)
        limits = self.data.get_limits()
        self.shape = limits[1] - limits[0]

        if periodic is not None:
            self.set_periodic(periodic)

        logger.debug("Lattice shape: %s (%s)", self.shape,
                     frmt_num(self.data.nbytes, unit="iB", div=1024))
        return self.data

    def _build_periodic_segment(self, indices, positions, axs, out_ind=None,
                                out_pos=None):
        limits = np.array([np.min(indices, axis=0), np.max(indices, axis=0)])
        idx_size = (limits[1] - limits[0])[:-1]
        nvec = np.zeros_like(idx_size, dtype=np.int64)
        for ax in np.atleast_1d(axs):
            nvec[ax] = np.floor(idx_size[ax]) + 1

        delta_pos = self.translate(nvec)
        delta_idx = np.append(nvec, 0)
        if out_ind is not None and out_pos is not None:
            out_ind[:] = indices + delta_idx
            out_pos[:] = positions + delta_pos
            return nvec

        out_ind = indices + delta_idx
        out_pos = positions + delta_pos
        return nvec, out_ind, out_pos

    def _compute_periodic_neighbors(self, indices, positions, axis, num_jobs=-1):
        axis = np.atleast_1d(axis)
        invald_idx = len(indices)

        # Build tree
        k = np.sum(np.sum(self._raw_num_neighbors, axis=1)) + 1
        max_dist = np.max(self.distances) + 0.1 * np.min(self._raw_distance_matrix)
        tree = KDTree(positions, k, max_dist)

        # Initialize arrays
        ind_t = np.zeros_like(indices)
        pos_t = np.zeros_like(positions)

        pidx, pdists, paxs = dict(), dict(), dict()
        for ax in axis:
            # Translate positions along periodic axis
            self._build_periodic_segment(indices, positions, ax, out_ind=ind_t,
                                         out_pos=pos_t)

            # Query neighbors with translated points and filter
            neighbors, distances = tree.query(pos_t, num_jobs, self.DIST_DECIMALS)
            neighbors, distances = self._filter_neighbors(indices, neighbors,
                                                          distances, ind_t)

            # Convert to dict
            idx = np.where(np.isfinite(distances).any(axis=1))[0]
            distances = distances[idx]
            neighbors = neighbors[idx]
            for i, site in enumerate(idx):
                mask = i, neighbors[i] < invald_idx
                inds = neighbors[mask]
                dists = distances[mask]

                pidx.setdefault(site, list()).extend(inds)  # noqa
                pdists.setdefault(site, list()).extend(dists)
                paxs.setdefault(site, list()).extend([ax] * len(inds))
                for j, d in zip(inds, dists):
                    pidx.setdefault(j, list()).append(site)  # noqa
                    pdists.setdefault(j, list()).append(d)
                    paxs.setdefault(j, list()).append(ax)

        for k in pidx.keys():
            vals, ind = np.unique(pidx[k], return_index=True)
            pidx[k] = np.array(vals)
            pdists[k] = np.array(pdists[k])[ind]
            paxs[k] = np.array(paxs[k])[ind]

        return pidx, pdists, paxs

    def set_periodic(self, axis: Union[bool, int, Sequence[int]]):
        """Sets periodic boundary conditions along the given axis.

        Parameters
        ----------
        axis : bool or int or (N, ) array_like
            One or multiple axises to apply the periodic boundary conditions.
            If the axis is `None` the perodic boundary
            conditions will be removed.

        Raises
        ------
        NotBuiltError
            Raised if the lattice hasn't been built yet.

        Notes
        -----
        The lattice has to be built before applying the periodic boundarie conditions.
        The lattice also has to be at least three atoms big in the specified directions.
        """
        if isinstance(axis, bool) and axis is True:
            axis = np.arange(self.dim)

        logger.debug("Computing periodic neighbors along axis %s", axis)
        if self.shape is None:
            raise NotBuiltError()
        axis = np.atleast_1d(axis)

        indices = self.data.indices
        positions = self.data.positions
        pidx, pdists, paxs = self._compute_periodic_neighbors(indices, positions, axis)

        self.data.set_periodic(pidx, pdists, paxs)
        self.periodic_axes = axis

    def _compute_connection_neighbors(self, positions1, positions2):
        # Set neighbor query parameters
        k = np.sum(np.sum(self._raw_num_neighbors, axis=1)) + 1
        max_dist = np.max(self.distances) + 0.1 * np.min(self._raw_distance_matrix)

        # Build sub-lattice tree's
        tree1 = KDTree(positions1, k=k, max_dist=max_dist)
        tree2 = KDTree(positions2, k=k, max_dist=max_dist)

        pairs = list()
        distances = list()
        offset = len(positions1)
        connections = tree1.query_ball_tree(tree2, max_dist)
        for i, conns in enumerate(connections):
            if conns:
                conns = np.asarray(conns)
                dists = cdist(np.asarray([positions1[i]]), positions2[conns])[0]
                for j, dist in zip(conns + offset, dists):
                    pairs.append((i, j))
                    pairs.append((j, i))
                    distances.append(dist)
                    distances.append(dist)

        return np.array(pairs), np.array(distances)

    def append(self, latt, ax=0, side=+1, sort_axis=None, sort_reverse=False):
        # Build translation vector
        indices = self.data.indices if side > 0 else latt.data.indices
        nvec = build_periodic_translation_vector(indices, ax)
        if side <= 0:
            nvec = -1 * nvec
        vec = self.translate(nvec)

        # Store temporary data
        positions1 = self.data.positions
        indices2 = latt.data.indices.copy()
        positions2 = latt.data.positions.copy()
        neighbors2 = latt.data.neighbors.copy()
        distances2 = latt.data.distvals[latt.data.distances]

        # Shift data of appended lattice
        indices2[:, :-1] += nvec
        positions2 += vec

        # Append data and compute connecting neighbors
        self.data.append(indices2, positions2, neighbors2, distances2)
        pairs, distances = self._compute_connection_neighbors(positions1, positions2)
        for (i, j), dist in zip(pairs, distances):
            self.data.add_neighbors(i, j, dist)

        if sort_axis is not None:
            self.data.sort(sort_axis, reverse=sort_reverse)

        # Update the shape of the lattice
        limits = self.data.get_limits()
        self.shape = limits[1] - limits[0]

    def dmap(self) -> DataMap:
        """Returns the data-map of the lattice model."""
        return self.data.map()

    # ==================================================================================

    def copy(self) -> 'Lattice':
        """Lattice : Creates a (deep) copy of the lattice instance."""
        return deepcopy(self)

    def todict(self) -> dict:
        d = dict()
        d["vectors"] = self.vectors
        d["atoms"] = self._atoms
        d["positions"] = self._positions
        d["connections"] = self._connections
        d["shape"] = self.shape
        return d

    def dumps(self):
        lines = list()
        for key, values in self.todict().items():
            head = key + ":"
            lines.append(f"{head:<15}" + "; ".join(str(x) for x in values))
        return "\n".join(lines)

    def dump(self, file: Union[str, int, bytes] = 'tmp.latt') -> None:
        """Save the data of the `Lattice` instance.

        Parameters
        ----------
        file : str or int or bytes
            File name to store the lattice. If `None` the hash of the lattice is used.
        """
        if file is None:
            file = f"{self.__hash__()}.latt"
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file: Union[str, int, bytes] = 'tmp.latt') -> 'Lattice':
        """Load data of a saved `Lattice` instance.

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

    def plot_cell(self,
                  lw: float = 1.,
                  color: Union[str, float] = 'k',
                  alpha: float = 0.5,
                  legend: bool = True,
                  margins: Union[Sequence[float], float] = 0.25,
                  show_cell: bool = True,
                  show_vecs: bool = True,
                  show_neighbors: bool = True,
                  ax: Union[plt.Axes, Axes3D] = None,
                  show: bool = False) -> Union[plt.Axes, Axes3D]:
        """Plot the unit cell of the lattice.

        Parameters
        ----------
        lw : float, default: 1
            Line width of the hopping connections.
        color : str, optional
            Optional string for color of cell-lines.
        alpha : float, optional
            Optional alpha value of neighbors.
        legend : bool, optional
            Flag if legend is shown.
        margins : Sequence[float] or float, optional
            Optional margins of the plot.
        show_neighbors : bool, optional
            If `True` the neighbors are plotted.
        show_vecs : bool, optional
            If 'True' the first unit-cell is drawn.
        show_cell : bool, optional
            If `True` the outlines of the unit cell are plotted.
        ax : plt.Axes or plt.Axes3D or None, optional
            Parent plot. If None, a new plot is initialized.
        show : bool, optional
            If `True`, show the resulting plot.
        """
        if self.dim > 3:
            raise ValueError(f"Plotting in {self.dim} dimensions is not supported!")
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d" if self.dim == 3 else None)
        else:
            fig = ax.get_figure()

        # prefetch colors
        colors = list()
        for i in range(self.num_base):
            atom = self.get_atom(i)
            line = ax.plot([], [], color=atom.color)[0]
            col = line.get_color()
            colors.append(col)

        # Draw unit vectors and the cell they spawn.
        if show_vecs:
            vectors = self.vectors
            draw_cell(ax, vectors, color="k", lw=1., outlines=show_cell)

        if show_neighbors:
            position_arr = [list() for _ in range(self.num_base)]
            for i in range(self.num_base):
                pos = self.atom_positions[i]
                for distidx in range(self.num_distances):
                    try:
                        indices = self.get_neighbors(alpha=i, distidx=distidx)
                        positions = self.get_positions(indices)
                        draw_vectors(ax, positions - pos, pos=pos, zorder=1,
                                     color=color, lw=lw)
                        for idx, pos1 in zip(indices, positions):
                            if np.any(idx[:-1]):
                                a = idx[-1]
                                position_arr[a].append(pos1)
                    except IndexError:
                        pass

            for i in range(self.num_base):
                atom = self.get_atom(i)
                positions = position_arr[i]
                if positions:
                    pos = np.unique(positions, axis=0)
                    size = 0.6 * atom.size
                    col = colors[i]
                    draw_points(ax, pos, size=size, color=col, label=atom.name,
                                alpha=alpha)

        # Plot atoms in the unit cell
        for i in range(self.num_base):
            atom = self.get_atom(i)
            pos = self.atom_positions[i]
            col = colors[i]
            draw_points(ax, pos, size=atom.size, color=col, label=atom.name)

        # Format plot
        if legend and self._num_base > 1:
            ax.legend()
        if isinstance(margins, float):
            margins = [margins] * self.dim
        if self.dim == 1:
            w = self.cell_size[0]
            ax.set_ylim(-w / 2, +w / 2)
        else:
            ax.margins(*margins)

        if self.dim == 3:
            ax.set_aspect("equal")
        else:
            ax.set_aspect("equal", "box")

        fig.tight_layout()
        if show:
            plt.show()
        return ax

    def plot(self,
             lw: float = 1.,
             color: Union[str, float, int] = 'k',
             margins: Union[Sequence[float], float] = 0.1,
             legend: bool = True,
             grid: bool = False,
             show_periodic: bool = True,
             show_indices: bool = False,
             show_cell: bool = False,
             ax: Union[plt.Axes, Axes3D] = None,
             show: bool = False) -> Union[plt.Axes, Axes3D]:
        """Plot the cached lattice.

        Parameters
        ----------
        lw : float, default: 1
            Line width of the hopping connections.
        color : str or float or int
            Line color of the hopping connections.
        margins : Sequence[float] or float, optional
            Optional margins of the plot.
        legend : bool, optional
            Flag if legend is shown
        grid : bool, optional
            If 'True', draw a grid in the plot.
        show_periodic : bool, optional
            If 'True' the periodic connections will be shown.
        show_indices : bool, optional
            If 'True' the index of the sites will be shown.
        show_cell : bool, optional
            If 'True' the first unit-cell is drawn.
        ax : plt.Axes or plt.Axes3D or None, optional
            Parent plot. If None, a new plot is initialized.
        show : bool, optional
            If `True`, show the resulting plot.
        """
        logger.debug("Plotting lattice")

        if self.dim > 3:
            raise ValueError(f"Plotting in {self.dim} dimensions is not supported!")

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d" if self.dim == 3 else None)
        else:
            fig = ax.get_figure()

        # Draw unit vectors and the cell they spawn.
        if show_cell:
            vectors = self.vectors
            draw_cell(ax, vectors, color='k', lw=2, outlines=True)

        # Draw connections
        limits = self.data.get_translation_limits()
        idx_size = limits[1] - limits[0]
        nvecs_diag = np.floor(idx_size) + 1
        nvecs = np.diag(nvecs_diag)
        for i in range(self.num_sites):
            pos = self.data.positions[i]
            neighbor_pos = self.data.get_neighbor_pos(i, periodic=False)
            if len(neighbor_pos):
                draw_vectors(ax, neighbor_pos - pos, pos=pos, color=color, lw=lw,
                             zorder=1)
                if show_periodic:
                    mask = self.data.neighbor_mask(i, periodic=True)
                    idx = self.data.neighbors[i, mask]
                    paxes = self.data.paxes[i, mask]
                    neighbor_pos = self.data.positions[idx]
                    for pax, x in zip(paxes, neighbor_pos):
                        nvec = nvecs[pax]
                        sign = +1 if x[pax] < pos[pax] else -1
                        x = self.translate(sign * nvec, x)
                        draw_vectors(ax, x - pos, pos=pos, color=color, lw=lw, zorder=1)

        # Draw sites
        for alpha in range(self.num_base):
            atom = self.atoms[alpha]
            points = self.data.get_positions(alpha)
            draw_points(ax, points, size=atom.size, color=atom.color, label=atom.name)

        if show_indices:
            positions = [self.position(i) for i in range(self.num_sites)]
            draw_indices(ax, positions)

        # Format plot
        if legend and self._num_base > 1:
            ax.legend()
        if grid:
            ax.set_axisbelow(True)
            ax.grid(b=True, which='major')

        if isinstance(margins, float):
            margins = [margins] * self.dim
        if self.dim == 1 or (self.dim == 2 and self.shape[1] < 1):
            sizex = self.shape[0]
            h = sizex / 4
            ax.set_ylim(-h, +h)
        else:
            ax.margins(*margins)

        if self.dim == 3:
            ax.set_aspect("equal")
        else:
            ax.set_aspect("equal", "box")

        fig.tight_layout()
        if show:
            plt.show()
        return ax

    def __repr__(self) -> str:
        shape = str(self.shape) if self.shape is not None else "None"
        return f"{self.__class__.__name__}(dim: {self.dim}, " \
               f"num_base: {self.num_base}, shape: {shape})"
