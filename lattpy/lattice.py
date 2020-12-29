# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import pickle
import itertools
import collections
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from typing import Union, Optional, Tuple, List, Iterator, Sequence, Callable, Any, Dict

from .utils import vrange, SiteOccupiedError, NoAtomsError, NoBaseNeighboursError, NotBuiltError
from .plotting import draw_points, draw_vectors, draw_cell, draw_indices
from .spatial import WignerSeitzCell, KDTree, compute_neighbours, cell_size, cell_volume
from .unitcell import Atom
from .data import LatticeData


class Lattice:
    """Object representing the basis and data of a bravais lattice."""

    DIST_DECIMALS: int = 6        # Decimals used for rounding distances
    RVEC_TOLERANCE: float = 1e-6  # Tolerance for reciprocal vectors/lattice

    def __init__(self, vectors: Union[float, Sequence[float], Sequence[Sequence[float]]],
                 **kwargs):
        """Initialize a new ``Lattice`` instance.

        Parameters
        ----------
        vectors: array_like or float
            The vectors that span the basis of the lattice.
        """
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

        # Neighbour data
        self._num_distances = 0
        self._num_neighbours = None
        self._base_neighbours = None
        self._distances = None

        # Lattice Cache
        self.data = LatticeData()
        self.shape = None
        self.periodic_axes = list()

    @classmethod
    def chain(cls, a: Optional[float] = 1.0, **kwargs) -> 'Lattice':
        return cls(a, **kwargs)

    @classmethod
    def square(cls, a: Optional[float] = 1.0, **kwargs) -> 'Lattice':
        return cls(a * np.eye(2), **kwargs)

    @classmethod
    def rectangular(cls, a1: Optional[float] = 1., a2: Optional[float] = 1., **kwargs) -> 'Lattice':
        return cls(np.array([[a1, 0], [0, a2]]), **kwargs)

    @classmethod
    def hexagonal(cls, a: Optional[float] = 1.0, **kwargs) -> 'Lattice':
        vectors = a / 2 * np.array([[3, np.sqrt(3)], [3, -np.sqrt(3)]])
        return cls(vectors, **kwargs)

    @classmethod
    def oblique(cls, alpha: float, a1: Optional[float] = 1.0,
                a2: Optional[float] = 1.0, **kwargs) -> 'Lattice':
        vectors = np.array([[a1, 0], [a2 * np.cos(alpha), a2 * np.sin(alpha)]])
        return cls(vectors, **kwargs)

    @classmethod
    def hexagonal3D(cls, a: Optional[float] = 1., az: Optional[float] = 1., **kwargs) -> 'Lattice':  # noqa
        vectors = a / 2 * np.array([[3, np.sqrt(3), 0], [3, -np.sqrt(3), 0], [0, 0, az]])
        return cls(vectors, **kwargs)

    @classmethod
    def sc(cls, a: Optional[float] = 1.0, **kwargs) -> 'Lattice':
        return cls(a * np.eye(3), **kwargs)

    @classmethod
    def fcc(cls, a: Optional[float] = 1.0, **kwargs) -> 'Lattice':
        vectors = a/2 * np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        return cls(vectors, **kwargs)

    @classmethod
    def bcc(cls, a: Optional[float] = 1.0, **kwargs) -> 'Lattice':
        vectors = a/2 * np.array([[1, 1, 1], [1, -1, 1], [-1, 1, 1]])
        return cls(vectors, **kwargs)

    # ==============================================================================================

    def copy(self) -> 'Lattice':
        """ Creates a (deep) copy of the lattice instance"""
        return deepcopy(self)

    @property
    def dim(self) -> int:
        """The dimension of the vector basis."""
        return self._dim

    @property
    def vectors(self) -> np.ndarray:
        """Array with basis vectors as rows"""
        return self._vectors.T

    @property
    def vectors3d(self) -> np.ndarray:
        """Basis vectors expanded to three dimensions """
        vectors = np.eye(3)
        vectors[:self.dim, :self.dim] = self._vectors
        return vectors.T

    @property
    def norms(self):
        """Lengths of the basis-vectors"""
        return np.linalg.norm(self._vectors, axis=0)

    @property
    def cell_size(self) -> np.ndarray:
        """The shape of the box spawned by the given vectors."""
        return self._cell_size

    @property
    def cell_volume(self) -> float:
        """The volume of the unit cell defined by the primitive vectors."""
        return self._cell_volume

    @property
    def num_base(self) -> int:
        """The number of atoms in the unitcell."""
        return self._num_base

    @property
    def atoms(self) -> List[Atom]:
        """List of the atoms in the unitcell."""
        return self._atoms

    @property
    def atom_positions(self) -> List[np.ndarray]:
        """List of corresponding positions of the atoms in the unitcell."""
        return self._positions

    @property
    def num_distances(self):
        """The number of distances between the lattice sites."""
        return self._num_distances

    @property
    def num_neighbours(self):
        """The number of neighbours of each atom in the unitcell."""
        return self._num_neighbours

    @property
    def base_neighbours(self):
        """The neighbours of the unitcell at the origin."""
        return self._base_neighbours

    @property
    def distances(self) -> List[float]:
        """List of distances between the lattice sites."""
        return self._distances

    @property
    def num_sites(self) -> int:
        """Number of sites in lattice data (only available if lattice has been built)."""
        return self.data.num_sites

    @property
    def num_cells(self) -> int:
        """Number of unit-cells in lattice data (only available if lattice has been built)."""
        return np.unique(self.data.indices[:, :-1], axis=0).shape[0]

    def transform(self, world_coords: Union[Sequence[int], Sequence[Sequence[int]]]) -> np.ndarray:
        """ Transform the world-coordinates (x, y, ...) into the basis coordinates (n, m, ...)

        Parameters
        ----------
        world_coords: (..., N) array_like

        Returns
        -------
        basis_coords: (..., N) np.ndarray
        """
        world_coords = np.atleast_1d(world_coords)
        if len(world_coords.shape) == 1:
            return np.asarray(world_coords) @ self._vectors_inv
        else:
            return np.dot(world_coords, self._vectors_inv[np.newaxis, :, :])[:, 0, :]

    def itransform(self, basis_coords: Union[Sequence[int], Sequence[Sequence[int]]]) -> np.ndarray:
        """ Transform the basis-coordinates (n, m, ...) into the world coordinates (x, y, ...)

        Parameters
        ----------
        basis_coords: (..., N) array_like

        Returns
        -------
        world_coords: (..., N) np.ndarray
        """
        basis_coords = np.atleast_1d(basis_coords)
        if len(basis_coords.shape) == 1:
            return basis_coords @ self._vectors
        else:
            return np.dot(basis_coords, self.vectors[np.newaxis, :, :])[:, 0, :]

    def translate(self, nvec: Union[int, Sequence[int], Sequence[Sequence[int]]],
                  r: Optional[Union[float, Sequence[float]]] = 0.0) -> np.ndarray:
        r""" Translates the given postion vector r by the translation vector n.

        The position is calculated using the translation vector .math`n` and the
        atom position in the unitcell .math:`r`:
        ..math::
            R = \sum_i n_i v_i + r

        Parameters
        ----------
        nvec: (..., N) array_like
            Translation vector in the lattice basis.
        r: (N) array_like, optional
            The position in cartesian coordinates. If no vector is passed only
            the translation is returned.

        Returns
        -------
        r_trans: (N) np.ndarray
        """
        r = np.atleast_1d(r)
        nvec = np.atleast_1d(nvec)
        if len(nvec.shape) == 1:
            return r + (self._vectors @ nvec)
        else:
            return r + np.dot(nvec, self.vectors[np.newaxis, :, :])[:, 0, :]

    def itranslate(self, x: Union[float, Sequence[float]]) -> [np.ndarray, np.ndarray]:
        """ Returns the translation vector and atom position of the given position.

        Parameters
        ----------
        x: (N) array_like or float
            Position vector in cartesian coordinates.

        Returns
        -------
        nvec: (N) np.ndarray
            Translation vector in the lattice basis.
        r: (N) np.ndarray, optional
            The position in real-space.
        """
        x = np.atleast_1d(x)
        itrans = self._vectors_inv @ x
        nvec = np.floor(itrans)
        r = x - self.translate(nvec)
        return nvec, r

    def is_reciprocal(self, vecs: Union[float, Sequence[float], Sequence[Sequence[float]]],
                      tol: Optional[float] = RVEC_TOLERANCE) -> bool:
        r""" Checks if the given vectors are reciprocal to the lattice vectors.

        The lattice- and reciprocal vectors .math'a_i' and .math'b_i' must satisfy the relation
        ..math::
            a_i \cdot b_i = 2 \pi \delta_{ij}

        To check the given vectors, the difference of each dot-product is compared to
        .math:'2\pi' with the given tolerance.

        Parameters
        ----------
        vecs: array_like or float
            The vectors to check. Must have the same dimension as the lattice.
        tol: float, optional
            The tolerance used for checking the result of the dot-products.

        Returns
        -------
        is_reciprocal: bool
        """
        vecs = np.atleast_2d(vecs)
        two_pi = 2 * np.pi
        for a, b in zip(self.vectors, vecs):
            if abs(np.dot(a, b) - two_pi) > tol:
                return False
        return True

    def reciprocal_vectors(self, tol: Optional[float] = RVEC_TOLERANCE,
                           check: Optional[bool] = False) -> np.ndarray:
        r""" Computes the reciprocal basis vectors of the bravais lattice.

        The lattice- and reciprocal vectors .math'a_i' and .math'b_i' must satisfy the relation
        ..math::
            a_i \cdot b_i = 2 \pi \delta_{ij}

        Parameters
        ----------
        tol: float, optional
            The tolerance used for checking the result of the dot-products.
        check: bool, optional
            Check the result and raise an exception if it doesn't satisfy the definition.
        Returns
        -------
        v_rec: np.ndarray
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

    def reciprocal_lattice(self, min_negative: Optional[bool] = False) -> 'Lattice':
        """Creates the lattice in reciprocal space

        Parameters
        ----------
        min_negative: bool, optional
            If 'True' the reciprocal vectors are scaled such that
            there are fewer negative elements than positive ones.

        Returns
        -------
        rlatt: Lattice
        """
        rvecs = self.reciprocal_vectors(min_negative)
        rlatt = self.__class__(rvecs)
        return rlatt

    def get_neighbour_cells(self, distidx: Optional[int] = 0,
                            include_origin: Optional[bool] = True,
                            comparison: Optional[Callable] = np.isclose) -> np.ndarray:
        """ Find all neighbouring unit cells.

        Parameters
        ----------
        distidx: int, default
            Index of distance to neighbouring cells, default is 0 (nearest neighbours).
        include_origin: bool, optional
            If ``True`` the origin is included in the set.
        comparison: callable, optional
            The method used for comparing distances.

        Returns
        -------
        indices: np.ndarray
        """
        # Build cell points
        max_factor = distidx + 1
        axis_factors = np.arange(-max_factor, max_factor + 1)
        factors = np.array(list(itertools.product(axis_factors, repeat=self.dim)))
        points = np.dot(factors, self.vectors[np.newaxis, :, :])[:, 0, :]

        # Compute distances to origin for all points
        distances = np.linalg.norm(points, axis=1)

        # Set maximum distances value to number of neighbours
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
        ws_cell: WignerSeitzCell
        """
        nvecs = self.get_neighbour_cells(include_origin=True)
        positions = np.dot(nvecs, self.vectors[np.newaxis, :, :])[:, 0, :]
        return WignerSeitzCell(positions)

    def brillouin_zone(self, min_negative: Optional[bool] = False) -> WignerSeitzCell:
        """Computes the first Brillouin-zone of the lattice structure.

        Constructs the Wigner-Seitz cell of the reciprocal lattice

        Parameters
        ----------
        min_negative: bool, optional
            If 'True' the reciprocal vectors are scaled such that
            there are fewer negative elements than positive ones.

        Returns
        -------
        ws_cell: WignerSeitzCell
        """
        rvecs = self.reciprocal_vectors(min_negative)
        rlatt = self.__class__(rvecs)
        return rlatt.wigner_seitz_cell()

    def check_points(self, points: np.ndarray,
                     shape: Union[int, Sequence[int]],
                     relative: Optional[bool] = False,
                     pos: Optional[Union[float, Sequence[float]]] = None
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
            The default is ``True``.
        pos: (N) array_like or int, optional
            Optional position of the section to build. If 'None' the origin is used.

        Returns
        -------
        mask: (M) np.ndarray
        """
        shape = np.atleast_1d(shape)
        if len(shape) != self.dim:
            raise ValueError(f"Dimension of shape {len(shape)} doesn't "
                             f"match the dimension of the lattice {self.dim}")
        if relative:
            shape = np.array(shape) * np.max(self.vectors, axis=0) - 0.1 * self.norms

        if pos is None:
            pos = np.zeros(self.dim)
        end = pos + shape

        mask = (pos[0] <= points[:, 0]) & (points[:, 0] <= end[0])
        for i in range(1, self.dim):
            mask = mask & (pos[i] <= points[:, i]) & (points[:, i] <= end[i])
        return mask

    def filter_points(self, points: np.ndarray,
                      shape: Union[int, Sequence[int]],
                      relative: Optional[bool] = False,
                      pos: Optional[Union[float, Sequence[float]]] = None
                      ) -> np.ndarray:
        """Returns the points that are in the given shape.

        Parameters
        ----------
        points: (M, N) np.ndarray
            The points in cartesian coordinates.
        shape: (N) array_like or int
            shape of finite size lattice to build.
        relative: bool, optional
            If 'True' the shape will be multiplied by the cell size of the model.
            The default is ``True``.
        pos: (N) array_like or int, optional
            Optional position of the section to build. If 'None' the origin is used.

        Returns
        -------
        points: (K, N) np.ndarray
        """
        mask = self.check_points(points, shape, relative, pos)
        return points[mask]

    def build_translation_vectors(self, shape: Union[int, Sequence[int]],
                                  relative: Optional[bool] = False,
                                  pos: Optional[Union[float, Sequence[float]]] = None,
                                  check: Optional[bool] = True
                                  ) -> np.ndarray:
        """Constructs the translation vectors .math:`n` in the lattice basis in a given shape.

        Raises
        ------
        ValueError
            Raised if the dimension of the position doesn't match the dimension of the lattice.

        Parameters
        ----------
        shape: (N) array_like or int
            shape of finite size lattice to build.
        relative: bool, optional
            If 'True' the shape will be multiplied by the cell size of the model.
            The default is ``True``.
        pos: (N) array_like or int, optional
            Optional position of the section to build. If 'None' the origin is used.
        check: bool, optional
            If ``True`` the positions of the translation vectors are checked and filtered.
            The default is ``True``. This should only be disabled if filtered later.

        Returns
        -------
        nvecs: (M, N) np.ndarray
            The translation-vectors in lattice-coordinates
        """
        shape = np.atleast_1d(shape)
        if len(shape) != self.dim:
            raise ValueError(f"Dimension of shape {len(shape)} doesn't "
                             f"match the dimension of the lattice {self.dim}")
        if relative:
            shape = np.array(shape) * np.max(self.vectors, axis=0) - 0.1 * self.norms
        if pos is None:
            pos = np.zeros(self.dim)
        end = pos + shape

        # Generate translation vectors with too many points.
        min_values = (self.itranslate(pos)[0] - shape).astype("int")
        max_values = (np.abs(self.itranslate(end)[0]) + shape).astype("int")
        min_values[min_values == 0] = -1  # set minimum size to 1
        max_values[max_values == 0] = +1  # set minimum size to 1

        ranges = [(range(min_values[d], max_values[d])) for d in range(self.dim)]
        nvecs = np.array(vrange(ranges))

        if check:
            # Filter points in the given volume
            positions = np.dot(nvecs, self.vectors[np.newaxis, :, :])[:, 0, :]
            mask = (pos[0] <= positions[:, 0]) & (positions[:, 0] <= end[0])
            for i in range(1, self.dim):
                mask = mask & (pos[i] <= positions[:, i]) & (positions[:, i] <= end[i])
            nvecs = nvecs[mask]
        return nvecs

    # ==============================================================================================

    def add_atom(self, pos: Optional[Union[float, Sequence[float]]] = None,
                 atom: Optional[Union[str, Dict[str, Any], Atom]] = None,
                 relative: Optional[bool] = False,
                 neighbours: Optional[int] = 0,
                 **kwargs) -> Atom:
        """ Adds a site to the basis of the lattice unit-cell.

        Raises
        ------
        ValueError
            Raised if the dimension of the position doesn't match the dimension of the lattice.
        ConfigurationError
            Raised if the position of the new atom is already occupied.

        Parameters
        ----------
        pos: (N) array_like or float, optional
            Position of site in the unit-cell. The default is the origin of the cell.
            The size of the array has to match the dimension of the lattice.
        atom: str or dict or Atom, optional
            Identifier of the site. If a string is passed, a new Atom instance is created.
        relative: bool, optional
            Flag if the specified position is in cartesian or lattice coordinates.
            If ``True`` the passed position will be multiplied with the lattice vectors.
            The default is ``False`` (cartesian coordinates).
        neighbours: int, optional
            The number of neighbor distance to calculate. If the number is ´0´ the distances have
            to be calculated manually after configuring the lattice basis.
        **kwargs
            Keyword arguments for ´Atom´ constructor. Only used if a new Atom instance is created.

        Returns
        -------
        atom: Atom
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
        assert len(self._atoms) == len(self._positions)
        self._num_base = len(self._positions)

        if neighbours:
            self.set_num_neighbours(neighbours)
        return atom

    def set_num_neighbours(self, num_neighbours: int = 1, analyze: bool = True) -> None:
        """ Sets the maximal neighbour distance of the lattice.

        Parameters
        ----------
        num_neighbours: int, optional
            The number of neighbour-distance levels,
            e.g. setting to `1` means only nearest neighbours.
        analyze: bool
            Flag if lattice base is analyzed. If `False` the `analyze`-method
            needs to be called manually. The default is `True`.
        """
        self._num_distances = num_neighbours
        if analyze:
            self.analyze()

    def _compute_base_neighbours(self, max_distidx, num_jobs=1):
        cell_range = 2 * max_distidx
        nvecs = self.get_neighbour_cells(cell_range, include_origin=True, comparison=np.less_equal)
        arrays = [np.c_[nvecs, i * np.ones(nvecs.shape[0])] for i in range(self.num_base)]
        cols = self.dim + 1
        indices = np.ravel(arrays, order="F").astype(np.int)
        indices = indices.reshape(cols, int(indices.shape[0] / cols)).T
        nvecs, alphas = indices[:, :-1], indices[:, -1]
        positions = self.translate(nvecs, np.array(self.atom_positions)[alphas])

        tree = KDTree(positions, k=len(positions))
        # Compute number of neighbours for each distance level

        base_neighbours = list()
        for alpha in range(self.num_base):
            pos = self.atom_positions[alpha]
            dists, idx = tree.query(pos, n_jobs=num_jobs)
            dists = np.round(dists, decimals=self.DIST_DECIMALS)
            neighbour_indices = indices[idx]
            # Store neighbours of certain distance
            neighbours = collections.OrderedDict()
            for dist, idx in zip(dists, neighbour_indices):
                if dist:
                    neighbours.setdefault(dist, list()).append(idx)
            base_neighbours.append(neighbours)
        return base_neighbours

    def analyze(self, num_distances: Optional[int] = None) -> None:
        """ Analyzes the strucutre of the lattice and stores neighbour data of the unitcell.

        Checks distances between all sites of the bravais lattice and saves n lowest values.
        The neighbor lattice-indices of the unit-cell are also stored for later use.
        This speeds up many calculations like finding nearest neighbours.

        Raises
        ------
        NoAtomsError
            Raised if no atoms where added to the lattice. The atoms in the unit cell are needed
            for computing the neighbours and distances of the lattice.

        Parameters
        ----------
        num_distances: int, optional
            Number of nearest distances of the lattice structure to calculate.
            By default the previously set number of distances is used.

        """
        if len(self._atoms) == 0:
            raise NoAtomsError()

        if num_distances is None:
            num_distances = self.num_distances
        else:
            num_distances = max(num_distances, self.num_distances)
        base_neighbours = self._compute_base_neighbours(num_distances)
        # Cleanup data and convert to np.ndarray
        for alpha in range(self.num_base):
            neighbours = base_neighbours[alpha]
            dists = list(neighbours.keys())
            max_distidx = self.num_distances  # self._num_distances[alpha]
            for dist in dists[:max_distidx]:
                base_neighbours[alpha][dist] = np.asarray(neighbours[dist])
            for dist in dists[max_distidx:]:
                del base_neighbours[alpha][dist]

        # Compute number of neighbours for each atom in the unit cell
        num_neighbours = np.zeros(self.num_base, dtype=np.int8)
        for i, neighbours in enumerate(base_neighbours):
            num_neighbours[i] = sum(len(indices) for indices in neighbours.values())

        # store distance values / keys:
        distances = np.zeros((self.num_base, self.num_distances))  # max(self.num_distances)))
        for alpha in range(self.num_base):
            try:
                dists = list(base_neighbours[alpha].keys())
            except ValueError:
                dists = list()
            distances[alpha, :len(dists)] = dists

        self._base_neighbours = base_neighbours
        self._num_neighbours = num_neighbours
        self._distances = distances

    def calculate_distances(self, num_dist: Optional[int] = 1) -> None:
        """ Calculates the ´n´ lowest distances between sites and the neighbours of the cell.

        Notes
        -----
        Deprecated: Use `set_num_neighbours` instead.
        """
        self.set_num_neighbours(num_dist, analyze=True)

    def get_position(self, nvec: Optional[Union[int, Sequence[int]]] = None,
                     alpha: Optional[int] = 0) -> np.ndarray:
        """ Returns the position for a given translation vector and site index

        Parameters
        ----------
        nvec: (N) array_like or int
            translation vector.
        alpha: int, optional
            site index, default is 0.
        Returns
        -------
        pos: (N) np.ndarray
        """
        r = self._positions[alpha]
        if nvec is None:
            return r
        n = np.atleast_1d(nvec)
        return r + (self._vectors @ n)  # self.translate(n, r)

    def get_positions(self, indices):
        """Returns the positions for multiple lattice indices

        Parameters
        ----------
        indices: (N, D+1) array_like or int
            List of lattice indices.

        Returns
        -------
        pos: (N, D) np.ndarray
        """
        nvecs, alphas = indices[:, :-1], indices[:, -1]
        return self.translate(nvecs, np.array(self.atom_positions)[alphas])

    def get_alpha(self, atom: Union[int, str, Atom]) -> int:
        """Returns the index of the atom in the unit-cell.

        Parameters
        ----------
        atom: int or str or Atom
            The argument for getting the atom. If a ``int`` is passed
            it is interpreted as the index, if a ``str`` is passed as
            the name of an atom.

        Returns
        -------
        alpha: int
        """
        if isinstance(atom, Atom):
            return self._atoms.index(atom)
        elif isinstance(atom, str):
            for i, at in enumerate(self._atoms):
                if atom == at.name:
                    return i
        return atom

    def get_atom(self, atom: Union[int, str, Atom]) -> Atom:
        """ Returns the Atom object of the given atom in the unit cell

        Parameters
        ----------
        atom: int or str or Atom
            The argument for getting the atom. If a ``int`` is passed
            it is interpreted as the index, if a ``str`` is passed as
            the name of an atom.

        Returns
        -------
        atom: Atom
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

    def get_atom_attrib(self, atom: Union[int, str, Atom], attrib: str,
                        default: Optional[Any] = None) -> Any:
        """ Returns an attribute of a specific atom in the unit cell.

        Parameters
        ----------
        atom: int or str or Atom
            The argument for getting the atom. If a ``int`` is passed
            it is interpreted as the index, if a ``str`` is passed as
            the name of an atom.
        attrib: str
            Name of the atom attribute.
        default: str or int or float or object, optional
            Default value used if the attribute couln't be found in the Atom dictionary.

        Returns
        -------
        attrib: str or int or float or object
        """
        atom = self.get_atom(atom)
        return atom.get(attrib, default)

    def estimate_index(self, pos: Union[float, Sequence[float]]) -> np.ndarray:
        """ Returns the nearest matching lattice index (n, alpha) for global position.

        Parameters
        ----------
        pos: array_like or float
            global site position.

        Returns
        -------
        n: np.ndarray
            estimated translation vector n
        """
        pos = np.asarray(pos)
        n = np.asarray(np.round(self._vectors_inv @ pos, decimals=0), dtype="int")
        return n

    def get_neighbours(self, nvec: Optional[Union[int, Sequence[int]]] = None,
                       alpha: Optional[int] = 0,
                       distidx: Optional[int] = 0) -> np.ndarray:
        """ Returns the neighour-indices of a given site by transforming stored neighbour indices.

        Raises
        ------
        NoBaseNeighboursError
            Raised if the lattice distances and base-neighbours haven't been computed.

        Parameters
        ----------
        nvec: (D) array_like or int, optional
            translation vector of site, the default is the origin.
        alpha: int, optional
            site index, default is 0.
        distidx: int, default
            index of distance to neighbours, defauzlt is 0 (nearest neighbours).

        Returns
        -------
        indices: (N, D) np.ndarray
        """
        if nvec is None:
            nvec = np.zeros(self.dim)
        if not self._base_neighbours:
            raise NoBaseNeighboursError()

        nvec = np.atleast_1d(nvec)
        keys = list(sorted(self._base_neighbours[alpha].keys()))
        dist = keys[distidx]
        indices = self._base_neighbours[alpha][dist]
        indices_transformed = indices.copy()
        indices_transformed[:, :-1] += nvec.astype(np.int)
        return indices_transformed

    def get_neighbour_positions(self, nvec: Optional[Union[int, Sequence[int]]] = None,
                                alpha: Optional[int] = 0,
                                distidx: Optional[int] = 0) -> np.ndarray:
        """Returns the neighour-positions of a given site by transforming the neighbour positions.

        Raises
        ------
        NoBaseNeighboursError
            Raised if the lattice distances and base-neighbours haven't been computed.

        Parameters
        ----------
        nvec: (D) array_like or int, optional
            translation vector of site, the default is the origin.
        alpha: int, optional
            site index, default is 0.
        distidx: int, default
            index of distance to neighbours, defauzlt is 0 (nearest neighbours).

        Returns
        -------
        positions: (N, D) np.ndarray
        """
        if nvec is None:
            nvec = np.zeros(self.dim)
        if not self._base_neighbours:
            raise NoBaseNeighboursError()
        indices = self.get_neighbours(nvec, alpha, distidx)
        nvecs, alphas = indices[:, :-1], indices[:, -1]
        atom_pos = self._positions[alphas]
        return self.translate(nvecs, atom_pos)

    def get_neighbour_vectors(self, alpha: Optional[int] = 0,
                              distidx: Optional[int] = 0,
                              include_zero: Optional[bool] = False) -> List[np.ndarray]:
        """ Returns the neighours of a given site by transforming stored neighbour indices.

        Raises
        ------
        NoBaseNeighboursError
            Raised if the lattice distances and base-neighbours haven't been computed.

        Parameters
        ----------
        alpha: int, optional
            Index of the base atom. The default is the first atom in the unit cell.
        distidx: int, default
            Index of distance to neighbours, default is 0 (nearest neighbours).
        include_zero: bool, optional
            Flag if zero-vector is included in result. The default is False.

        Returns
        -------
        vectors: list of np.ndarray
        """
        if not self._base_neighbours:
            raise NoBaseNeighboursError()
        pos0 = self._positions[alpha]
        pos1 = self.get_neighbour_positions(alpha=alpha, distidx=distidx)
        if include_zero:
            pos1 = np.append(np.zeros((1, self.dim)), pos1, axis=0)
        return pos1 - pos0

    def get_base_atom_dict(self, atleast2d: Optional[bool] = True) \
            -> Dict[Any, List[Union[np.ndarray, Any]]]:
        """ Returns a dictionary containing the positions for eatch type of the base atoms.

        Parameters
        ----------
        atleast2d: bool, optional
            If 'True', one-dimensional coordinates will be casted to 2D vectors.

        Returns
        -------
        atom_pos: dict
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

    def build_indices(self, shape: Union[int, Sequence[int]],
                      relative: Optional[bool] = False,
                      pos: Optional[Union[float, Sequence[float]]] = None,
                      check: Optional[bool] = True,
                      callback: Optional[Callable] = None,
                      dtype: Union[int, np.dtype] = np.int,
                      ) -> np.ndarray:
        """Constructs the lattice indices .math:`(n, \alpha)` in the given shape.

        Raises
        ------
        ValueError
            Raised if the dimension of the position doesn't match the dimension of the lattice.

        Parameters
        ----------
        shape: (N) array_like or int
            shape of finite size lattice to build.
        relative: bool, optional
            If 'True' the shape will be multiplied by the cell size of the model.
            The default is ``True``.
        pos: (N) array_like or int, optional
            Optional position of the section to build. If 'None' the origin is used.
        check: bool, optional
            If ``True`` the positions of the translation vectors are checked and filtered.
            The default is ``True``. This should only be disabled if filtered later.
        callback: callable, optional
            Optional callable for filtering positions.
        dtype: int or np.dtype, optional
            Optional data-type for storing the lattice indices. The default is ``np.int``.
            Using a smaller bit-size may help reduce memory usage.

        Returns
        -------
        indices: (M, N+1) np.ndarray
            The lattice indices of the sites in the format .math:`(n_1, .. n_d, \alpha)`.
        """
        nvecs = self.build_translation_vectors(shape, relative, pos, check=False)
        arrays = [np.c_[nvecs, i * np.ones(nvecs.shape[0])] for i in range(self.num_base)]
        cols = self.dim + 1
        indices = np.ravel(arrays, order="F")
        indices = indices.reshape(cols, int(indices.shape[0] / cols)).T
        # Filter points in the given volume
        positions = [self.translate(nvecs, pos) for pos in self.atom_positions]
        positions = np.ravel(positions, order="F")
        positions = positions.reshape(self.dim, int(positions.shape[0] / self.dim)).T
        if check:
            mask = self.check_points(positions, shape, relative, pos)
            indices = indices[mask]
            positions = positions[mask]
        if callback is not None:
            indices = indices[callback(positions)]
        return indices.astype(dtype=dtype)

    def compute_neighbours(self, positions: Union[Sequence[float], Sequence[Sequence[float]]],
                           num_jobs: Optional[int] = 1) -> Tuple[np.ndarray, np.ndarray]:
        """ Computes the neighbours for the given points.

        Parameters
        ----------
        positions: array_like
            An array of points to compute the neighbours.
        num_jobs: int, optional
            Number of jobs to schedule for parallel processing.
            If -1 is given all processors are used. The default is ``1``.

        Returns
        -------
        neighbours: (..., M) np.ndarray
            The indices of the neighbours in ``positions``.
        distances: (..., M) np.ndarray
            The corresponding distances of the neighbours.
        """
        max_dist = np.max(self.distances) + 0.1 * np.min(self.distances)
        k = np.max(self.num_neighbours) + 1
        idx, dists = compute_neighbours(positions, k=k, max_dist=max_dist, num_jobs=num_jobs)
        dists = np.round(dists, decimals=self.DIST_DECIMALS)
        return idx, dists

    # ==============================================================================================
    # Cached lattice
    # ==============================================================================================

    def volume(self) -> float:
        """The total volume (number of cells x cell-volume) of the buildt lattice."""
        return self.cell_volume * np.unique(self.data.indices[:, :-1], axis=0).shape[0]

    def alpha(self, idx: int) -> int:
        """Returns the atom component of the lattice index of the given site

        Parameters
        ----------
        idx: int
            Site index in the cached lattice data.

        Returns
        -------
        alpha: int
        """
        return self.data.indices[idx, -1]

    def atom(self, idx: int) -> Atom:
        """ Returns the atom of a given site in the cached lattice data.

        Parameters
        ----------
        idx: int
            Site index in the cached lattice data.

        Returns
        -------
        atom: Atom
        """
        return self._atoms[self.data.indices[idx, -1]]

    def position(self, idx: int) -> np.ndarray:
        """ Returns the position for a given site in the cached lattice data.

        Parameters
        ----------
        idx: int
            Site index in the cached lattice data.

        Returns
        -------
        pos: (N) np.ndarray
        """
        return self.data.positions[idx]

    def neighbours(self, site: int, distidx: Optional[int] = None,
                   unique: Optional[bool] = False) -> np.ndarray:
        """ Returns the neighours of a given site in the cached lattice data.

        Parameters
        ----------
        site: int
            Site index in the cached lattice data.
        distidx: int, default
            Index of distance to neighbours, defauzlt is 0 (nearest neighbours).
        unique: bool, optional
            If 'True', each unique pair is only return once.

        Returns
        -------
        indices: np.ndarray of int
        """
        return self.data.get_neighbours(site, distidx, unique=unique)

    def nearest_neighbours(self, idx: int, unique: Optional[bool] = False) -> np.ndarray:
        """ Returns the nearest neighours of a given site in the cached lattice data.

        Parameters
        ----------
        idx: int
            Site index in the cached lattice data.
        unique: bool, optional
            If 'True', each unique pair is only return once.

        Returns
        -------
        indices: np.ndarray of int
        """
        return self.neighbours(idx, 0, unique)

    def iter_neighbours(self, site: int,
                        unique: Optional[bool] = False) -> Iterator[Tuple[int, np.ndarray]]:
        """Iteratse over the neighbours of all distances of a given site in the cached lattice data.

        Parameters
        ----------
        site: int
            Site index in the cached lattice data.
        unique: bool, optional
            If 'True', each unique pair is only return once.


        Yields
        -------
        distidx: int
        neighbours: np.ndarray
        """
        return self.data.iter_neighbours(site, unique)

    def check_neighbours(self, idx0: int, idx1: int) -> Union[float, None]:
        """ Checks if two sites are neighbours and returns the distance-idx if they are.

        Parameters
        ----------
        idx0: int
            First site index in the cached lattice data.
        idx1: int
            Second site index in the cached lattice data.

        Returns
        -------
        distidx: int or None
        """
        for distidx in range(self.num_distances):
            if idx1 in self.neighbours(idx0, distidx):
                return distidx
        return None

    def build(self, shape: Union[int, Sequence[int]],
              relative: Optional[bool] = False,
              pos: Optional[Union[float, Sequence[float]]] = None,
              check: Optional[bool] = True,
              num_jobs: Optional[int] = 1,
              periodic: Optional[Union[int, Sequence[int]]] = None,
              dtype: Union[int, np.dtype] = np.int,
              callback: Optional[Callable] = None) -> LatticeData:
        """ Constructs the indices and neighbours of a new finite size lattice and stores the data

        Raises
        ------
        ValueError
            Raised if the dimension of the position doesn't match the dimension of the lattice.
        NoBaseNeighboursError
            Raised if the lattice distances and base-neighbours haven't been computed.

        Parameters
        ----------
        shape: (N) array_like or int
            shape of finite size lattice to build.
        relative: bool, optional
            If 'True' the shape will be multiplied by the cell size of the model.
            The default is ``True``.
        pos: (N) array_like or int, optional
            Optional position of the section to build. If 'None' the origin is used.
        check: bool, optional
            If ``True`` the positions of the translation vectors are checked and filtered.
            The default is ``True``. This should only be disabled if filtered later.
        num_jobs: int, optional
            Number of jobs to schedule for parallel processing of neighbours.
            If -1 is given all processors are used. The default is ``1``.
        periodic: int or array_like, optional
            Optional periodic axes to set. See 'set_periodic' for mor details.
        dtype: int or np.dtype, optional
            Optional data-type for storing the lattice indices. The default is ``np.int``.
            Using a smaller bit-size may help reduce memory usage.
        callback: callable

        """
        self.data.reset()
        shape = np.atleast_1d(shape)
        if len(shape) != self.dim:
            raise ValueError(f"Dimension of shape {len(shape)} doesn't "
                             f"match the dimension of the lattice {self.dim}")
        if not self._base_neighbours:
            raise NoBaseNeighboursError()

        # Build indices and positions
        indices = self.build_indices(shape, relative, pos, check, callback, dtype)
        positions = self.get_positions(indices)

        # Compute the neighbours and distances between the sites
        neighbours, distances = self.compute_neighbours(positions, num_jobs=num_jobs)

        # Set data of the lattice
        self.data.set(indices, positions, neighbours, distances)
        limits = self.data.get_limits()
        self.shape = limits[1] - limits[0]

        if periodic is not None:
            self.set_periodic(periodic)
        return self.data

    def _build_periodic_segment(self, indices, ax):
        limits = np.array([np.min(indices, axis=0), np.max(indices, axis=0)])
        idx_size = (limits[1] - limits[0])[:-1]
        nvec = np.zeros_like(idx_size, dtype=np.int)
        nvec[ax] = np.floor(idx_size[ax]) + 1

        delta = self.num_distances  # self.max_distidx
        idx_size[ax] = delta

        nvec1 = nvec + idx_size
        ranges = [(range(nvec[d], nvec1[d] + 1)) for d in range(self.dim)]
        nvecs = np.array(vrange(ranges))

        arrays = [np.c_[nvecs, i * np.ones(nvecs.shape[0])] for i in range(self.num_base)]
        cols = self.dim + 1
        indices2 = np.ravel(arrays, order="F").astype(np.int)
        indices2 = indices2.reshape(cols, int(indices2.shape[0] / cols)).T
        positions2 = self.get_positions(indices2)
        return nvec, indices2, positions2

    def _compute_periodic_neighbours(self, indices, positions, ax):
        max_dist = np.max(self.distances)
        k = np.max(self.num_neighbours) + 3

        periodic_neighbours = dict()
        periodic_distances = dict()
        periodic_axes = dict()
        for ax in np.atleast_1d(ax):
            nvec, indices2, positions2 = self._build_periodic_segment(indices, ax)
            neighbours, distances = compute_neighbours(positions2, positions, k, max_dist,
                                                       num_jobs=1)
            distances = np.round(distances, decimals=self.DIST_DECIMALS)
            idx = np.where(np.isfinite(distances).any(axis=1))[0]
            distances = distances[idx]
            neighbours = neighbours[idx]
            periodic_indices = indices[idx]
            periodic_indices[:, ax] -= nvec[ax] - 1

            num_points = len(positions2)
            for i, site in enumerate(idx):
                mask = neighbours[i] < num_points  # noqa
                dists = distances[i][mask]
                neighbour_indices = indices2[neighbours[i][mask]]  # noqa
                neighbour_indices[:, :-1] -= nvec
                sites2 = [np.where(np.array(indices == x).all(axis=1))[0][0] for x in
                          neighbour_indices]

                periodic_neighbours.setdefault(site, list()).extend(sites2)
                periodic_distances.setdefault(site, list()).extend(dists)
                periodic_axes.setdefault(site, list()).append(ax)
                for j, d in zip(sites2, dists):
                    periodic_neighbours.setdefault(j, list()).append(site)
                    periodic_distances.setdefault(j, list()).append(d)
                    periodic_axes.setdefault(j, list()).append(ax)

        return periodic_neighbours, periodic_distances, periodic_axes

    def set_periodic(self, axis: Optional[Union[int, Sequence[int]]] = 0):
        """ Sets periodic boundary conditions along the given axis.

        Notes
        -----
        The lattice has to be built before applying the periodic boundarie conditions.
        Also the lattice has to be at least three atoms big in the specified directions.

        Raises
        ------
        NotBuiltError
            Raised if the lattice hasn't been built yet.

        Parameters
        ----------
        axis: int or (N) array_like, optional
            One or multiple axises to apply the periodic boundary conditions.
            The default is the x-direction. If the axis is `None` the perodic boundary
            conditions will be removed.
        """
        if self.shape is None:
            raise NotBuiltError()
        axis = np.atleast_1d(axis)

        indices = self.data.indices
        positions = self.data.positions
        pidx, pdists, paxs = self._compute_periodic_neighbours(indices, positions, axis)

        self.data.set_periodic(pidx, pdists, paxs)
        self.periodic_axes = axis

    def transform_periodic(self, pos: Union[Union[float, Sequence[float]]],
                           ax: Union[int, Sequence[int]],
                           cell_offset: Optional[float] = 0.0) -> np.ndarray:
        """ Transforms the position along the given axis by the shape of the lattice.

        Raises
        ------
        NotBuiltError
            Raised if the lattice hasn't been built yet.

        Parameters
        ----------
        pos: array_like
            Position coordinates to tranform.
        ax: int or array_like
            The axis that the position will be transformed along.
        cell_offset: float, optional
            Optional offset in units of the cell size.

        Returns
        -------
        transformed: np.ndarray
        """
        if self.shape is None:
            raise NotBuiltError()
        pos = np.atleast_1d(pos)
        delta = np.zeros(self.dim, dtype="float")
        # Get cell offset along axis
        delta[ax] = self.cell_size[ax] * cell_offset
        # Get translation along axis
        delta[ax] += self.shape[ax]
        # Get transformation direction (to nearest periodic cell)
        sign = -1 if pos[ax] > self.shape[ax] / 2 else +1
        return pos + sign * delta

    def atom_positions_dict(self, indices: Optional[Sequence[Sequence[int]]] = None,
                            atleast2d: Optional[bool] = True) -> Dict[Atom, List[np.ndarray]]:
        """ Returns a dictionary containing the positions for each type of the atoms.

        Raises
        ------
        NotBuiltError
            Raised if the lattice hasn't been built yet.

        Parameters
        ----------
        indices: array_like, optional
            Optional indices to use. If 'None' the stored indices are used.
        atleast2d: bool, optional
            If 'True', one-dimensional coordinates will be casted to 2D vectors.
        Returns
        -------
        atom_pos: dict
        """
        if self.shape is None:
            raise NotBuiltError()

        indices = self.data.indices if indices is None else indices
        atom_pos = dict()
        for idx in indices:
            n, alpha = idx[:-1], idx[-1]
            atom = self._atoms[alpha]
            pos = self.get_position(n, alpha)
            if atleast2d and self.dim == 1:
                pos = np.array([pos, 0])

            if atom.name in atom_pos.keys():
                atom_pos[atom].append(pos)
            else:
                atom_pos[atom] = [pos]
        return atom_pos

    def all_positions(self) -> np.ndarray:
        """ Returns all positions, independent of the atom type, for the lattice.

        Raises
        ------
        NotBuiltError
            Raised if the lattice hasn't been built yet.

        Returns
        -------
        positions: array_like
        """
        if self.shape is None:
            raise NotBuiltError()
        return np.asarray([self.position(i) for i in range(self.num_sites)])

    def get_connections(self, atleast2d: Optional[bool] = True) -> np.ndarray:
        """ Returns all pairs of neighbours in the lattice


        Raises
        ------
        NotBuiltError
            Raised if the lattice hasn't been built yet.

        Parameters
        ----------
        atleast2d: bool, optional
            If 'True', one-dimensional coordinates will be casted to 2D vectors.

        Returns
        -------
        connections: array_like
        """
        if self.shape is None:
            raise NotBuiltError()
        conns = list()
        for i in range(self.num_sites):
            # neighbor_list = self.data.neighbours[i]
            for distidx in range(self.num_distances):
                # for j in neighbor_list[distidx]:
                neighbours = self.data.get_neighbours(i, distidx, periodic=False)
                for j in neighbours:
                    if j > i:
                        p1 = self.position(i)
                        p2 = self.position(j)
                        if atleast2d and self.dim == 1:
                            p1 = np.array([p1, 0])
                            p2 = np.array([p2, 0])
                        conns.append([p1, p2])
        return np.asarray(conns)

    def get_periodic_segments(self, scale: Optional[float] = 1.0,
                              atleast2d: Optional[bool] = True
                              ) -> List[List[np.ndarray]]:
        """ Returns all pairs of peridoic neighbours in the lattice

        Raises
        ------
        NotBuiltError
            Raised if the lattice hasn't been built yet.

        Parameters
        ----------
        scale: float, optional
        atleast2d: bool, optional
            If 'True', one-dimensional coordinates will be casted to 2D vectors.

        Returns
        -------
        connections: array_like
        """
        if self.shape is None:
            raise NotBuiltError()
        conns = list()
        for i in range(int(self.num_sites)):
            p1 = self.position(i)
            for distidx in range(self.num_distances):
                neighbours = self.data.get_neighbours(i, distidx, periodic=True)
                # neighbours = self.data.periodic_neighbours[i][distidx]
                if neighbours:
                    for j in neighbours:
                        p2_raw = self.position(j)
                        # Find cycling axis
                        ax = np.argmax(np.abs(p2_raw - p1))
                        # Transform back
                        p2 = self.transform_periodic(p2_raw, ax, cell_offset=1.0)
                        # Scale vector and add to connections
                        if atleast2d and self.dim == 1:
                            p1 = np.array([p1, 0])
                            p2 = np.array([p2, 0])
                        v = p2 - p1
                        p2 = p1 + scale * v / np.linalg.norm(v)
                        conns.append([p1, p2])
        return conns

    # ==============================================================================================

    def save(self, file: Optional[Union[str, int, bytes]] = 'tmp_lattice.pkl') -> None:
        """Save the data of the ``Lattice`` instance.

        Parameters
        ----------
        file: str or int or bytes
        """
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file: Optional[Union[str, int, bytes]] = 'tmp_lattice.pkl') -> 'Lattice':
        """Load data of a saved ``Lattice`` instance.

        Parameters
        ----------
        file: str or int or bytes

        Returns
        -------
        latt: Lattice
        """
        with open(file, "rb") as f:
            latt = pickle.load(f)
        return latt

    def plot_cell(self, show: Optional[bool] = True,
                  ax: Optional[Union[plt.Axes, Axes3D]] = None,
                  lw: Optional[float] = 1.,
                  color: Optional[Union[str, float]] = 'k',
                  alpha: Optional[float] = 0.5,
                  legend: Optional[bool] = True,
                  margins: Optional[Union[Sequence[float], float]] = 0.25,
                  show_cell: Optional[bool] = True,
                  show_vecs: Optional[bool] = True,
                  show_neighbours: Optional[bool] = True) -> Union[plt.Axes, Axes3D]:
        """ Plot the unit cell of the lattice.

        Parameters
        ----------
        show: bool, default: True
            parameter for pyplot.
        ax: plt.Axes or plt.Axes3D or None, optional
            Parent plot. If None, a new plot is initialized.
        lw: float, default: 1
            Line width of the hopping connections.
        color: str, optional
            Optional string for color of cell-lines.
        alpha: float, optional
            Optional alpha value of neighbours.
        legend: bool, optional
            Flag if legend is shown.
        margins: Sequence[float] or float, optional
            Optional margins of the plot.
        show_neighbours: bool, optional
            If ``True`` the neighbours are plotted.
        show_vecs: bool, optional
            If 'True' the first unit-cell is drawn.
        show_cell: bool, optional
            If ``True`` the outlines of the unit cell are plotted.
        """
        if self.dim > 3:
            raise ValueError(f"Plotting in {self.dim} dimensions is not supported!")
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d" if self.dim == 3 else None)
        else:
            fig = ax.get_figure()

        # Draw unit vectors and the cell they spawn.
        if show_vecs:
            vectors = self.vectors
            draw_cell(ax, vectors, color="k", lw=1., outlines=show_cell)

        if show_neighbours:
            for i in range(self.num_base):
                pos = self.atom_positions[i]
                for distidx in range(self.num_distances):
                    indices = self.get_neighbours(alpha=i, distidx=distidx)
                    positions = self.get_positions(indices)
                    draw_vectors(ax, positions - pos, pos=pos, zorder=1, color=color, lw=lw)
                    for idx, pos1 in zip(indices, positions):
                        if np.any(idx[:-1]):
                            atom = self.atoms[idx[-1]]
                            draw_points(ax, pos1, size=atom.size * 0.75,
                                        color=atom.color, alpha=alpha)

        # Plot atoms in the unit cell
        for i in range(self.num_base):
            atom = self.get_atom(i)
            pos = self.atom_positions[i]
            draw_points(ax, pos, size=atom.size, color=atom.color, label=atom.name)

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

    def plot(self, show: Optional[bool] = True,
             ax: Optional[Union[plt.Axes, Axes3D]] = None,
             lw: Optional[float] = 1.,
             color: Optional[Union[str, float, int]] = 'k',
             margins: Optional[Union[Sequence[float], float]] = 0.1,
             legend: Optional[bool] = True,
             grid: Optional[bool] = False,
             show_periodic: Optional[bool] = True,
             show_indices: Optional[bool] = False,
             show_cell: Optional[bool] = False) -> Union[plt.Axes, Axes3D]:
        """Plot the cached lattice.

        Parameters
        ----------
        show: bool, default: True
            parameter for pyplot
        ax: plt.Axes or plt.Axes3D or None, optional
            Parent plot. If None, a new plot is initialized.
        lw: float, default: 1
            Line width of the hopping connections.
        color: str or float or int
            Line color of the hopping connections.
        margins: Sequence[float] or float, optional
            Optional margins of the plot.
        legend: bool, optional
            Flag if legend is shown
        grid: bool, optional
            If 'True', draw a grid in the plot.
        show_periodic: bool, optional
            If 'True' the periodic connections will be shown.
        show_indices: bool, optional
            If 'True' the index of the sites will be shown.
        show_cell: bool, optional
            If 'True' the first unit-cell is drawn.
        """
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
            neighbour_pos = self.data.get_neighbour_pos(i, periodic=False)
            draw_vectors(ax, neighbour_pos - pos, pos=pos, color=color, lw=lw, zorder=1)
            if show_periodic:
                mask = self.data.neighbour_mask(i, periodic=True)
                idx = self.data.neighbours[i, mask]
                paxes = self.data.paxes[i, mask]
                neighbour_pos = self.data.positions[idx]
                for pax, x in zip(paxes, neighbour_pos):
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
        if self.dim == 1:
            sizex = self.shape[0]
            h = sizex / 4
            w = self.cell_size[0]
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
        shape = str(self.shape) if self.shape else "None"
        return f"{self.__class__.__name__}(dim: {self.dim}, " \
               f"num_base: {self.num_base}, shape: {shape})"
