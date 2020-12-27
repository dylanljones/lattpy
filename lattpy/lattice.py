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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from typing import (
    Union, Optional, Tuple, List, Iterator, Sequence, Callable,
    Any, Dict, Iterable, Set
)
from .utils import (
    vrange, distance, cell_size, cell_volume,
    SiteOccupiedError, NoAtomsError, NoBaseNeighboursError, NotBuiltError
)
from .plotting import (
    draw_cell, draw_sites, draw_lines, draw_indices, set_padding
)
from .unitcell import Atom
from .data import LatticeData
from .geometry import WignerSeitzCell


class Lattice:
    """Main object representing the basis and data of a bravais lattice."""

    DIST_DECIMALS: int = 5       # Decimals used for rounding distances
    REC_TOLERANCE: float = 1e-5  # Tolerance for reciprocal vectors/lattice
    MIN_DISTS: int = 3           # Minimum distances that are computed

    def __init__(self, vectors: Union[Union[float, Sequence[float]],
                                      Sequence[Union[float, Sequence[float]]]],
                 atom: Optional[Union[str, Atom]] = None,
                 pos: Optional[Union[float, Sequence[float]]] = None,
                 neighbours: Optional[int] = 1,
                 **atom_kwargs):
        """Initialize a new ``Lattice`` instance.

        Parameters
        ----------
        vectors: (N, N) array_like or float
            The vectors that span the basis of the lattice.
        pos: (N) array_like or float, optional
            Position of site in the unit-cell. The default is the origin of the cell.
            The size of the array has to match the dimension of the basis vectors.
            If ``pos`` or ``atom`` arguments a passed the ``add_atom``-method is called.
        atom: str or Atom, optional
            Identifier of the site. If a string is passed, a new ``Atom`` instance is created.
            If ``pos`` or ``atom`` arguments a passed the ``add_atom``-method is called.
        neighbours: int, optional
            The number of neighbor distance to calculate. If the number is ``0`` the distances have
            to be calculated manually after configuring the lattice basis.
        **atom_kwargs
            Keyword arguments for ``Atom`` constructor. Only used if a new Atom instance is created.
        """
        # Vector basis
        self._vectors = np.atleast_2d(vectors).T
        self._vectors_inv = np.linalg.inv(self._vectors)
        self._dim = len(self._vectors)
        self._cell_size = cell_size(self.vectors)
        self._cell_volume = cell_volume(self.vectors)

        # Atom data
        self._num_base = 0
        self._num_dist = 0
        self._atoms = list()
        self._positions = list()
        self._distances = list()
        self._base_neighbors = list()

        # Lattice Cache
        self.data = LatticeData()
        self.shape = None
        self.periodic_axes = list()

        self.modifiers = list()

        # Quick setup
        if atom is not None or pos is not None:
            self.add_atom(pos=pos, atom=atom, neighbours=neighbours, **atom_kwargs)

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

    def copy(self) -> 'Lattice':
        """ Creates a (deep) copy of the lattice instance"""
        latt = self.__class__(self._vectors.copy().T)
        if self._num_base:
            latt._num_base = self._num_base
            latt._num_dist = self._num_dist
            latt._atoms = self._atoms.copy()
            latt._positions = self._positions.copy()
            latt._distances = self._distances.copy()
            latt._base_neighbors = self._base_neighbors.copy()
        if self.data:
            latt.shape = self.shape.copy()
            latt.data = self.data.copy()
            latt.periodic_axes = self.periodic_axes.copy()
        return latt

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
    def num_dist(self) -> int:
        """The number of distances between the lattice sites computed."""
        return self._num_dist

    @property
    def distances(self) -> List[float]:
        """List of distances between the lattice sites."""
        return self._distances

    @property
    def atoms(self) -> List[Atom]:
        """List of the atoms in the unitcell."""
        return self._atoms

    @property
    def atom_positions(self) -> List[np.ndarray]:
        """List of corresponding positions of the atoms in the unitcell."""
        return self._positions

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
        """ Translates the given postion vector r by the translation vector n.

        Parameters
        ----------
        nvec: (..., N) array_like
            Translation vector in the lattice basis.
        r: (N) array_like, optional
            The position in real-space. If no vector is passed only the translation is returned.

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

    def itranslate(self, v: Union[float, Sequence[float]]) -> [np.ndarray, np.ndarray]:
        """ Returns the lattice index and cell position leading to the given position in real space.

        Parameters
        ----------
        v: (N) array_like or float
            Position vector in real-space.

        Returns
        -------
        nvec: (N) np.ndarray
            Translation vector in the lattice basis.
        r: (N) np.ndarray, optional
            The position in real-space.
        """
        v = np.atleast_1d(v)
        itrans = self._vectors_inv @ v
        nvec = np.floor(itrans)
        r = v - self.translate(nvec)
        return nvec, r

    def is_reciprocal(self, vecs: Union[Union[float, Sequence[float]],
                                        Sequence[Union[float, Sequence[float]]]],
                      tol: Optional[float] = REC_TOLERANCE) -> bool:
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

    def reciprocal_vectors(self, tol: Optional[float] = REC_TOLERANCE,
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
        """Constructs the translation vectors .math:`n` in thhe lattice basis in a given shape.

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

    def translate_cell(self, nvec: Union[int, Sequence[int]]) -> np.ndarray:
        """ Translates all sites of the unit cell

        Parameters
        ----------
        nvec: np.ndarray
            translation vector.

        Yields
        -------
        pos: np.ndarray
            positions of the sites in the translated unit cell
        """
        nvec = np.atleast_1d(nvec)
        for alpha in range(self._num_base):
            yield self.get_position(nvec, alpha)

    def distance(self, idx0: Tuple[Sequence[int], int], idx1: Tuple[Sequence[int], int],
                 decimals: Optional[int] = DIST_DECIMALS) -> float:
        """ Calculate distance between two sites

        Parameters
        ----------
        idx0: tuple
            lattice vector (n, alpha) of first site
        idx1: tuple
            lattice index (n, alpha) of second site
        decimals: int, optional
            Optional decimals to round distance to.

        Returns
        -------
        distance: float
        """
        r1 = self.get_position(*idx0)
        r2 = self.get_position(*idx1)
        return distance(r1, r2, decimals)

    def _neighbour_cell_range(self, nvec: Optional[Union[int, Sequence[int]]] = None,
                              cell_range: Optional[int] = 1) -> Iterable[np.ndarray]:
        """ Get all neighbouring translation vectors of a given cell position

        Parameters
        ----------
        nvec: array_like, optional
            translation vector of unit cell, the default is the origin.
        cell_range: int, optional
            Range of neighbours, the default is 1.

        Returns
        -------
        nvecs: Iterable
        """
        nvec = np.zeros(self.dim) if nvec is None else np.atleast_1d(nvec)
        offset = cell_range + 2
        ranges = [np.arange(nvec[d] - offset, nvec[d] + offset + 1) for d in range(self.dim)]
        return vrange(ranges)

    def _neighbour_range(self, nvec: Optional[Union[int, Sequence[int]]] = None,
                         cell_range: Optional[int] = 1) -> [np.ndarray, int]:
        """ Get all neighbouring translation vectors and sublattice indices of a given cell position

        Parameters
        ----------
        nvec: array_like, optional
            translation vector of unit cell, the default is the origin.
        cell_range: int, optional
            Range of neighbours, the default is 1.

        Returns
        -------
        trans_vectors: list
        """
        nvec = np.zeros(self.dim) if nvec is None else np.atleast_1d(nvec)
        offset = cell_range + 2
        ranges = [np.arange(nvec[d] - offset, nvec[d] + offset + 1) for d in range(self.dim)]
        n_vecs = vrange(ranges)
        for n in n_vecs:
            for alpha in range(self._num_base):
                yield n, alpha

    def calculate_neighbours(self, nvec: Optional[Union[int, Sequence[int]]] = None,
                             alpha: Optional[int] = 0,
                             distidx: Optional[int] = 0,
                             array: Optional[bool] = False) -> List[Tuple[Sequence[int], int]]:
        """ Find all neighbours of given site and return the lattice indices.

        Raises
        ------
        NoAtomsError
            Raised if no atoms where added to the lattice.
            The atoms in the unit cell are needed for computing the distances in the lattice.

        Parameters
        ----------
        nvec: array_like, optional
            translation vector of site, the default is the origin.
        alpha: int, optional
            site index, default is 0.
        distidx: int, default
            index of distance to neighbours, default is 0 (nearest neighbours).
        array: bool, optional
            if true, return lattice index (n, alpha) as single array.
            The default is False.

        Returns
        -------
        indices: list
        """
        if len(self._atoms) == 0:
            raise NoAtomsError()
        nvec = np.zeros(self.dim) if nvec is None else np.atleast_1d(nvec)
        idx = nvec, alpha
        dist = self._distances[distidx]
        indices = list()
        for idx1 in self._neighbour_range(nvec, distidx):
            # if np.isclose(self.distance(idx, idx1), dist, atol=1e-5):
            if np.round(abs(self.distance(idx, idx1) - dist), decimals=self.DIST_DECIMALS) == 0.0:
                if array:
                    idx1 = [*idx1[0], idx1[1]]
                indices.append(idx1)
        return indices

    def calculate_distances(self, num_dist: Optional[int] = 1) -> None:
        """ alculates the ´n´ lowest distances between sites and the neighbours of the cell.

        Checks distances between all sites of the bravais lattice and saves n lowest values.
        The neighbor lattice-indices of the unit-cell are also stored for later use.
        This speeds up many calculations like finding nearest neighbours.

        Raises
        ------
        NoAtomsError
            Raised if no atoms where added to the lattice.
            The atoms in the unit cell are needed for computing the distances in the lattice.

        Parameters
        ----------
        num_dist: int, optional
            Number of distances of lattice structure to calculate.
            If 'None' the number of atoms is used.
            The default is 1 (nearest neighbours).
        """
        if len(self._atoms) == 0:
            raise NoAtomsError()

        if num_dist is None:
            num_dist = len(self._atoms)
        # Calculate n lowest distances of lattice structure
        n = max(num_dist, self.MIN_DISTS) + 1
        n_vecs = vrange(self.dim * [np.arange(-n, n)])
        r_vecs = list()
        for nvec in n_vecs:
            for alpha in range(self._num_base):
                r_vecs.append(self.get_position(nvec, alpha))
        pairs = list(itertools.product(r_vecs, self._positions))
        distances = list(set([distance(r1, r2, self.DIST_DECIMALS) for r1, r2 in pairs]))
        distances.sort()
        distances.remove(0.0)
        self._distances = distances[0:n - 1]
        self._num_dist = num_dist

        # Calculate cell-neighbors.
        neighbours = list()
        for alpha in range(self._num_base):
            site_neighbours = list()
            for i_dist in range(len(self._distances)):
                # Get neighbour indices of site for distance level
                new = self.calculate_neighbours(alpha=alpha, distidx=i_dist, array=True)
                site_neighbours.append(new)
            neighbours.append(site_neighbours)
        self._base_neighbors = neighbours

    def add_atom(self, pos: Optional[Union[float, Sequence[float]]] = None,
                 atom: Optional[Union[str, Dict[str, Any], Atom]] = None,
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
        neighbours: int, optional
            The number of neighbor distance to calculate. If the number is ´0´ the distances have
            to be calculated manually after configuring the lattice basis.
        **kwargs
            Keyword arguments for ´Atom´ constructor. Only used if a new Atom instance is created.

        Returns
        -------
        atom: Atom
        """
        if pos is None:
            pos = np.zeros(self.dim)
        else:
            pos = np.atleast_1d(pos)

        if len(pos) != self._dim:
            raise ValueError(f"Shape of the position {pos} doesn't match "
                             f"the dimension {self.dim} of the lattice!")

        if any(np.all(pos == x) for x in self._positions):
            raise SiteOccupiedError(atom, pos)

        if isinstance(atom, Atom):
            atom = atom
        else:
            atom = Atom(atom, **kwargs)

        self._atoms.append(atom)
        self._positions.append(np.asarray(pos))

        # Update number of base atoms if data is valid
        assert len(self._atoms) == len(self._positions)
        self._num_base = len(self._positions)

        if neighbours:
            self.calculate_distances(neighbours)
        return atom

    def get_neighbours(self, nvec: Optional[Union[int, Sequence[int]]] = None,
                       alpha: Optional[int] = 0,
                       distidx: Optional[int] = 0) -> List[np.ndarray]:
        """ Returns the neighour-indices of a given site by transforming stored neighbour indices.

        Raises
        ------
        NoBaseNeighboursError
            Raised if the lattice distances and base-neighbours haven't been computed.

        Parameters
        ----------
        nvec: array_like, optional
            translation vector of site, the default is the origin.
        alpha: int, optional
            site index, default is 0.
        distidx: int, default
            index of distance to neighbours, defauzlt is 0 (nearest neighbours).

        Returns
        -------
        indices: list of np.ndarray
        """
        if nvec is None:
            nvec = np.zeros(self.dim)
        if not self._base_neighbors:
            raise NoBaseNeighboursError()

        n = np.atleast_1d(nvec)
        transformed = list()
        for idx in self._base_neighbors[alpha][distidx]:
            idx_t = idx.copy()
            idx_t[:-1] += n
            transformed.append(idx_t)
        return transformed

    def get_neighbour_positions(self, nvec: Optional[Union[int, Sequence[int]]] = None,
                                alpha: Optional[int] = 0,
                                distidx: Optional[int] = 0) -> List[np.ndarray]:
        """Returns the neighour-positions of a given site by transforming the neighbour positions.

        Raises
        ------
        NoBaseNeighboursError
            Raised if the lattice distances and base-neighbours haven't been computed.

        Parameters
        ----------
        nvec: array_like, optional
            translation vector of site, the default is the origin.
        alpha: int, optional
            site index, default is 0.
        distidx: int, default
            index of distance to neighbours, defauzlt is 0 (nearest neighbours).

        Returns
        -------
        positions: list of np.ndarray
        """
        if nvec is None:
            nvec = np.zeros(self.dim)
        if not self._base_neighbors:
            raise NoBaseNeighboursError()

        n = np.atleast_1d(nvec)
        transformed = list()
        for idx in self._base_neighbors[alpha][distidx]:
            transformed.append(self.get_position(idx[:-1] + n, idx[-1]))
        return transformed

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
            Index of distance to neighbours, defauzlt is 0 (nearest neighbours).
        include_zero: bool, optional
            Flag if zero-vector is included in result. The default is False.

        Returns
        -------
        vectors: list of np.ndarray
        """
        if not self._base_neighbors:
            raise NoBaseNeighboursError()
        pos0 = self._positions[alpha]
        vectors = list()
        if include_zero:
            vectors.append(np.zeros(self.dim))
        for idx in self._base_neighbors[alpha][distidx]:
            pos1 = self.get_position(idx[:-1], idx[-1])
            vectors.append(pos1 - pos0)
        return vectors

    def get_neighbour_pairs(self, distidx: Optional[int] = 0) -> Iterator[Tuple[Union[float, int]]]:
        for alpha1 in range(self._num_base):
            pos0 = self.get_position(alpha=alpha1)
            for idx in self.get_neighbours(alpha=alpha1, distidx=distidx):
                n, alpha2 = idx[:-1], idx[-1]
                pos1 = self.get_position(n, alpha2)
                delta = pos1 - pos0
                yield delta, alpha1, alpha2

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

    # =========================================================================
    # Cached lattice
    # =========================================================================

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
        return self.data.indices[idx][-1]

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
        return self._atoms[self.data.indices[idx][-1]]

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
        # return self.get_position(*self.data.get_index(i))

    def neighbours(self, idx: int, distidx: Optional[int] = 0,
                   unique: Optional[bool] = False) -> Iterable[int]:
        """ Returns the neighours of a given site in the cached lattice data.

        Parameters
        ----------
        idx: int
            Site index in the cached lattice data.
        distidx: int, default
            Index of distance to neighbours, defauzlt is 0 (nearest neighbours).
        unique: bool, optional
            If 'True', each unique pair is only return once.

        Returns
        -------
        indices: list of int
        """
        if not hasattr(distidx, '__len__'):
            distidx = [distidx]
        neighbours = list()
        for di in distidx:
            neighbours += self.data.get_neighbours(idx, di)
        if unique:
            neighbours = [idx for idx in neighbours if idx > idx]
        return sorted(neighbours)

    def nearest_neighbours(self, idx: int, unique: Optional[bool] = False) -> Iterable[int]:
        """ Returns the nearest neighours of a given site in the cached lattice data.

        Parameters
        ----------
        idx: int
            Site index in the cached lattice data.
        unique: bool, optional
            If 'True', each unique pair is only return once.

        Returns
        -------
        indices: list of int
        """
        return self.neighbours(idx, 0, unique)

    def iter_neighbours(self, idx: int, unique: Optional[bool] = False) -> Iterator[Tuple[int]]:
        """ Iterate over the neighbours of all distances of a given site in the cached lattice data.

        Parameters
        ----------
        idx: int
            Site index in the cached lattice data.
        unique: bool, optional
            If 'True', each unique pair is only return once.

        Yields
        -------
        distidx: int
        siteidx: int
        """
        for distidx in range(self._num_dist):
            for j in self.neighbours(idx, distidx, unique):
                yield distidx, j

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
        for distidx in range(self._num_dist):
            if idx1 in self.neighbours(idx0, distidx):
                return distidx
        return None

    def iter_data(self):
        for i in range(self.num_sites):
            for distidx, j in self.iter_neighbours(i):
                yield i, j, distidx

    def _build_indices_inbound(self, shape: Union[float, Sequence[float]],
                               pos: Optional[Union[float, Sequence[float]]] = None
                               ) -> List[List[int]]:
        origin = np.zeros(self.dim)
        shape = np.atleast_1d(shape)
        if pos is None:
            pos = origin
        pos_idx = self.estimate_index(pos)
        # Find all indices that are in the volume
        max_values = np.abs(self.estimate_index(pos + shape))
        max_values[max_values == 0] = 1

        offset = 2 * shape
        offset[offset == 0] = 1
        ranges = list()
        for d in range(self.dim):
            x0, x1 = pos_idx[d] - offset[d], max_values[d] + offset[d]
            ranges.append(range(int(x0), int(x1)))
        nvecs = vrange(ranges)

        indices = list()
        for i, n in enumerate(nvecs):
            for alpha in range(self._num_base):
                # Check if index is in the real space shape
                include = True
                r = self.get_position(n, alpha)
                pos = origin if pos is None else pos
                for d in range(self.dim):
                    if not pos[d] <= r[d] <= pos[d] + shape[d]:
                        include = False
                        break
                if include:
                    indices.append([*n, alpha])
        return indices

    def _build_indices(self, shape: Union[float, Sequence[float]],
                       pos: Optional[Union[float, Sequence[float]]] = None) -> List[List[int]]:
        if pos is None:
            pos_idx = np.zeros(self.dim)
        else:
            pos_idx = self.estimate_index(pos)
        ranges = list()
        for d in range(self.dim):
            x0, x1 = pos_idx[d], pos_idx[d] + shape[d]
            ranges.append(range(int(x0), int(x1)))
        nvecs = vrange(ranges)
        indices = list()
        for i, n in enumerate(nvecs):
            for alpha in range(self._num_base):
                indices.append([*n, alpha])
        return indices

    def build_indices(self, shape: Union[int, Sequence[int]],
                      inbound: Optional[bool] = True,
                      cells: Optional[bool] = False,
                      pos: Optional[Union[float, Sequence[float]]] = None,
                      skip_indices: Optional[Union[int, Sequence[int]]] = None
                      ) -> List[List[int]]:
        """ Constructs the lattice indices in a given shape

        Raises
        ------
        ValueError
            Raised if the dimension of the position doesn't match the dimension of the lattice.

        Parameters
        ----------
        shape: array_like
            shape of finite size lattice to build.
        inbound: bool, optional
            If 'True' the shape will be interpreted in real-space. Only lattice-sites in this shape
            will be added to the data. This ensures nicer shapes of the lattice.
            Otherwise the shape is constructed in the basis if the unit-vectors.
            The default is 'True'
        cells: bool, optional
            If 'True' the shape will be multiplied by the cell size of the model.
            Only used if ``inbound==True``. The default is ``False``.
        pos: array_like, optional
            Optional position of the section to build. If 'None' the origin is used.
        skip_indices: array_like, optional
            Optional list of specific indices to skip.

        Returns
        -------
        indices: list of list of int
        """
        shape = np.atleast_1d(shape)
        if len(shape) != self.dim:
            raise ValueError(f"Dimension of shape {len(shape)} doesn't "
                             f"match the dimension of the lattice {self.dim}")
        if inbound:
            if cells:
                shape = np.array(shape) * np.max(self.vectors, axis=0) - 0.1
            indices = self._build_indices_inbound(shape, pos=pos)
        else:
            indices = self._build_indices(shape.astype('int'), pos=pos)
        if skip_indices:
            locs = list()
            for idx in np.atleast_2d(skip_indices):
                locs.extend(np.where((indices == np.array(idx)).all(axis=1))[0])
            indices = np.delete(indices, locs, axis=0)
        return indices

    def _construct(self, new_indices: Sequence[Sequence[int]],
                   new_neighbours: Optional[List[List[Set[Union[int, Any]]]]] = None,
                   site_indices: Optional[Sequence[Sequence[int]]] = None,
                   window: Optional[int] = None, buffer: Optional[int] = 5
                   ) -> [np.ndarray, np.ndarray, List[List[Set[Union[int, Any]]]]]:
        """ Constructs the index- and position-array and computes the neighbour indices.

        Raises
        ------
        NoBaseNeighboursError
            Raised if the lattice distances and base-neighbours haven't been computed.

        Parameters
        ----------
        new_indices: array_like
            Array of the new indices in the form of .math:'[n_1, .., n_N, alpha]'
            to add to the lattice. If the lattice doesn't have data yet a new array is created.
        new_neighbours: array_like, optional
            Optional array of new neighbours to add. by default a new array is created.
            This is used for adding new connections to an extisting lattice block.
        site_indices: array_like, optional
            Optional indices to calculate neighbours. This can be used for only computing neighbours
            in a certain region.
        window: int, optional
            Window for looking up neighbours. This can speed up the computation significally.
            Generally at least a few layers of the lattice should be searched.
            By default a window correspinding to 2 x atoms per slice x distances is used.
        buffer: int, optional
            Buffer used if no window is specified (smaller buffer means faster search).
            The default is ''3''.

        Returns
        -------
        indices: (N) np.ndarray
        positions: (N) np.ndarray
        neighbours: (N) list
        """
        num_sites = len(new_indices)
        n_dist = self._num_dist
        if not n_dist:
            raise NoBaseNeighboursError()

        # Construct indices and positions
        # -------------------------------

        # Initialize new indices and the empty neighbour array
        new_indices = np.array(new_indices)
        if new_neighbours is None:
            new_neighbours = [[set() for _ in range(n_dist)] for _ in range(num_sites)]

        # get all sites and neighbours (cached and new)
        if self.data:
            all_indices = np.append(self.data.indices, new_indices, axis=0)
            all_neighbours = self.data.neighbours[:] + new_neighbours
        else:
            all_indices = new_indices
            all_neighbours = new_neighbours

        # Compute position-vectors of the indices
        all_positions = np.zeros((len(all_indices), self.dim))
        for i, idx in enumerate(all_indices):
            all_positions[i] = self.get_position(idx[:-1], idx[-1])

        # Find neighbours
        # ---------------
        if window is None:
            # Estimate furthest index difference:
            # <Number of cells in biggest surface of shape>
            # x <number of base atoms>
            # x <number of distances>
            # x <buffer>
            limits = np.array([np.min(all_indices, axis=0), np.max(all_indices, axis=0)])
            shape = sorted((limits[1] - limits[0])[:self.dim], reverse=True)
            maxsurf = shape[0] * shape[1] if self.dim > 1 else shape[0]
            # maxdim = max(shape[1:]) if self.dim > 1 else 1
            window = buffer * self._num_dist * self._num_base * maxsurf

        # Find neighbours of each site in the "new_indices" list and store the neighbours
        offset = self.data.num_sites
        site_indices = site_indices if site_indices is not None else range(num_sites)
        for i in site_indices:
            site_idx = new_indices[i]
            n, alpha = np.array(site_idx[:-1]), site_idx[-1]
            i_site = i + offset

            # Get relevant index range to only look for neighbours
            # in proximity of site (larger than highest distance)
            i0 = max(i_site - window, 0)
            i1 = min(i_site + window, len(all_indices))
            win = np.arange(i0, i1)
            site_window = all_indices[win]

            # Get neighbour indices of site in proximity
            for i_dist in range(n_dist):
                # Get neighbour indices of site for distance level
                for idx in self.get_neighbours(n, alpha, i_dist):
                    # Find site of neighbour and store if in cache
                    hop_idx = np.where(np.all(site_window == idx, axis=1))[0]
                    if len(hop_idx):
                        j_site = hop_idx[0] + i0
                        all_neighbours[i_site][i_dist].add(j_site)
                        all_neighbours[j_site][i_dist].add(i_site)
            # all_neighbours = self._set_neighbours(site, idx, all_indices, all_neighbours, window)

        return all_indices, all_positions, all_neighbours

    def set_data(self, indices: np.ndarray,
                 positions: np.ndarray,
                 neighbours: List[List[Set[Union[int, Any]]]]) -> None:
        """ Sets cached data and re-computes real-space shape of lattice

        Parameters
        ----------
        indices: np.ndarray
            Lattice indices that will be saved.
        positions: np.ndarra
            Corresponding positions of the lattice indices.
        neighbours: list or list of set
            Neighbour indices of the lattice indices.
        """
        # Set data and recompute real-space shape of lattice
        self.data.set(indices, positions, neighbours)
        limits = self.data.get_limits()
        self.shape = limits[1] - limits[0]

    def build(self, shape: Union[int, Sequence[int]],
              inbound: Optional[bool] = True,
              cells: Optional[bool] = False,
              periodic: Optional[Union[int, Sequence[int]]] = None,
              pos: Optional[Union[float, Sequence[float]]] = None,
              window: Optional[int] = None,
              skip_indices: Optional[Union[int, Sequence[Sequence[int]]]] = None) -> None:
        """ Constructs the indices and neighbours of a new finite size lattice and stores the data

        Raises
        ------
        ValueError
            Raised if the dimension of the position doesn't match the dimension of the lattice.
        NoBaseNeighboursError
            Raised if the lattice distances and base-neighbours haven't been computed.

        Parameters
        ----------
        shape: array_like
            shape of finite size lattice to build.
        inbound: bool, optional
            If 'True' the shape will be interpreted in real-space. Only lattice-sites in this shape
            will be added to the data. This ensures nicer shapes of the lattice.
            Otherwise the shape is constructed in the basis if the unit-vectors.
            The default is 'True'
        cells: bool, optional
            If 'True' the shape will be multiplied by the cell size of the model.
            Only used if ``inbound==True``. The default is ``False``.
        periodic: int or array_like, optional
            Optional periodic axes to set. See 'set_periodic' for mor details.
        pos: array_like, optional
            Optional position of the section to build. If 'None' the origin is used.
        window: int, optional
            Window for looking up neighbours. This can speed up the computation significally.
            Generally at least a few layers of the lattice should be searched.
            By default a window correspinding to '2 x atoms per layer x nnumber
            of distances' is used.
        skip_indices: array_like, optional
            Optional list of specific indices to skip.
        """
        self.data.reset()
        shape = np.atleast_1d(shape)
        if len(shape) != self.dim:
            raise ValueError(f"Dimension of shape {len(shape)} doesn't "
                             f"match the dimension of the lattice {self.dim}")
        if not self._base_neighbors:
            raise NoBaseNeighboursError()

        # Compute indices and initialize neighbour array
        indices = self.build_indices(shape, inbound, cells, pos, skip_indices)
        indices, positions, neighbours = self._construct(indices, window=window)
        self.set_data(indices, positions, neighbours)
        if periodic is not None:
            self.set_periodic(periodic)

    def build_centered(self, shape:  Union[int, Sequence[int]]):
        """ Builds a centered lattice in the given (real world) coordinates.

        Parameters
        ----------
        shape: array_like
            shape of finite size lattice to build.
        """
        center = np.atleast_1d(shape) / 2
        self.build(shape, inbound=True, pos=-center)

    def set_periodic(self, axis: Optional[Union[int, Sequence[int]]] = 0):
        """ Sets periodic boundary conditions alogn the given axis.

        Adds the indices of the neighbours cycled around the given axis.

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
        self.data.neighbours.remove_all_periodic()
        if axis is None:
            # self.data.neighbours.remove_all_periodic()
            return
        axis = np.atleast_1d(axis)
        self.periodic_axes = axis
        n = int(self.num_sites)
        if self.dim == 1:
            for distidx in range(self._num_dist):
                i = distidx
                j = n - distidx - 1
                self.data.add_periodic_neighbour(i, j, distidx, symmetric=True)
        else:
            for ax in axis:
                # Get periodic translation vector
                vec = np.zeros(self.dim, dtype="float")
                vec[ax] = self.shape[ax] + 0.8 * self.cell_size[ax]
                nvec = vec @ np.linalg.inv(self._vectors.T)
                nvec[ax] = np.ceil(nvec[ax])
                nvec = np.round(nvec, decimals=0).astype("int")

                # Get window of outer sites along axis
                offset = 2 * self._num_dist * self.cell_size[ax]
                window = self.data.find_outer_sites(ax, offset)
                # window = np.arange(n)

                # Check if periodic neighbours
                for i in window:
                    pos1 = self.position(i)
                    for j in window:
                        pos2 = self.translate(nvec, self.position(j))
                        dist = distance(pos1, pos2, self.DIST_DECIMALS)
                        if dist in self._distances:
                            distidx = self._distances.index(dist)
                            if distidx < self._num_dist:
                                self.data.add_periodic_neighbour(i, j, distidx, symmetric=True)

    def add_x(self, lattice: 'Lattice', shift: Optional[bool] = True) -> None:
        n_new = lattice.num_sites
        new_data = lattice.data.copy()
        new_indices = new_data.indices
        new_neighbours = list()
        if shift:
            new_indices[:, 0] += self.estimate_index((self.shape[0], 0))[0] + 1
        for site_neighbours in new_data.neighbours:
            shifted = list()
            for dist_neighbours in site_neighbours:
                shifted.append(set([x + self.num_sites for x in dist_neighbours]))
            new_neighbours.append(shifted)
        # Find neighbours of connecting sections
        window = range(0, int(n_new / 2))
        indices, positions, neighbours = self._construct(new_indices, new_neighbours,
                                                         site_indices=window)
        # Set data
        self.set_data(indices, positions, neighbours)

    def __add__(self, other: 'Lattice') -> 'Lattice':
        new = self.copy()
        new.add_x(other)
        return new

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
            for distidx in range(self._num_dist):
                # for j in neighbor_list[distidx]:
                neighbours = self.data.get_nonperiodic_neighbours(i, distidx)
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
            for distidx in range(self._num_dist):
                neighbours = self.data.get_periodic_neighbours(i, distidx)
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

    # =========================================================================

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
                  color: Optional[str] = 'k',
                  alpha: Optional[float] = 0.5,
                  legend: Optional[bool] = True,
                  margins: Optional[float] = 0.25,
                  show_atoms: Optional[bool] = True,
                  show_neighbours: Optional[bool] = True,
                  show_cell: Optional[bool] = True,
                  outlines: Optional[bool] = True,
                  grid: Optional[bool] = True) -> Union[plt.Axes, Axes3D]:
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
        margins: float, optional
            Optional margins of the plot.
        show_atoms: bool, optional
            If ``True`` the atoms are plotted.
        show_neighbours: bool, optional
            If ``True`` the neighbours are plotted.
        show_cell: bool, optional
            If 'True' the first unit-cell is drawn.
        outlines: bool, optional
            If ``True`` the outlines of the unit cell are plotted.
        grid: bool, optional
            If 'True', draw a grid in the plot.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d" if self.dim == 3 else None)
        else:
            fig = ax.get_figure()

        # Plot atoms in the unit cell
        if show_atoms and self._num_base:
            atom_pos = self.get_base_atom_dict()
            for atom, positions in atom_pos.items():
                draw_sites(ax, positions, size=atom.size, color=atom.color, label=atom.name)

        # Draw neighbouring cell
        if show_neighbours:
            indices = list()
            segments = list()
            for alph in range(self.num_base):
                pos = self.get_position(alpha=alph)
                for distidx in range(self.num_dist):
                    for idx in self.get_neighbours(alpha=alph, distidx=distidx):
                        nvec = idx[:-1]
                        segments.append([pos, self.get_position(nvec, idx[-1])])
                        for _alpha in range(self.num_base):
                            indices.append([*nvec, _alpha])
            # indices = np.unique(indices, axis=0).astype(np.int)
            positions = dict()
            for idx in indices:
                nvec, alph = idx[:-1], idx[-1]
                pos = self.get_position(nvec, alph)
                positions.setdefault(alph, list()).append(pos)
                for distidx in range(self.num_dist):
                    for idx2 in self.get_neighbours(nvec, alph, distidx=distidx):
                        if len(np.where((indices == np.array(idx2)).all(axis=1))[0]):
                            nvec2, alpha2 = idx2[:-1], idx2[-1]
                            pos2 = self.get_position(nvec2, alpha2)
                            segments.append([pos, pos2])
            draw_lines(ax, segments, color="k", alpha=alpha, lw=lw)
            for alph, pos in positions.items():
                atom = self.atoms[alph]
                draw_sites(ax, pos, atom.size, alpha=alpha, color=atom.color)

        # Draw unit vectors and the cell they spawn.
        if show_cell:
            vectors = self.vectors
            draw_cell(ax, vectors, color, lw, outlines=outlines)

        # Format plot
        if self.dim == 1:
            w = self.cell_size[0]
            ax.set_xlim(-0.1*w, 1.1 * w)
            ax.set_ylim(-w/2, +w/2)
        elif self.dim == 3:
            ax.margins(margins, margins, margins)
        else:
            ax.margins(margins, margins)

        # Set equal aspect ratio
        if self.dim != 3:
            ax.set_aspect("equal", "box")

        if grid:
            ax.set_axisbelow(True)
            ax.grid()
        if legend and self._num_base:
            ax.legend()

        fig.tight_layout()
        if show:
            plt.show()
        return ax

    def plot(self, show: Optional[bool] = True,
             ax: Optional[Union[plt.Axes, Axes3D]] = None,
             lw: Optional[float] = 1.,
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
            Line width of the hopping connections
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
        # Prepare site positions and hopping segments
        atom_pos = self.atom_positions_dict()
        segments = self.get_connections()

        # Reuse or initialize new Plot
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d" if self.dim == 3 else None)
        else:
            fig = ax.get_figure()

        # Draw unit vectors and the cell they spawn.
        if show_cell:
            vectors = self.vectors
            draw_cell(ax, vectors, color='k', lw=2, outlines=True)

        # Draw atoms, neighbour connections and optionally site indices.
        draw_lines(ax, segments, color="k", lw=lw)
        if show_periodic and len(self.periodic_axes):
            scale = 0.3 * np.linalg.norm(self.vectors[0])
            periodic_segments = self.get_periodic_segments(scale=scale)
            draw_lines(ax, periodic_segments, color="0.5", lw=lw)

        for atom, positions in atom_pos.items():
            positions = np.array(positions)
            positions = positions[~np.isnan(positions).any(axis=1)]
            draw_sites(ax, positions, size=atom.size, color=atom.color, label=atom.name, alpha=1.0)
        if show_indices:
            positions = [self.position(i) for i in range(self.num_sites)]
            draw_indices(ax, positions)

        if self.dim == 1:
            w = self.cell_size[0]
            ax.set_ylim(-w, +w)
        elif self.dim == 2:
            set_padding(ax, self.cell_size[0]/2, self.cell_size[1]/2)
        else:
            ax.margins(0.1, 0.1, 0.1)

        if grid:
            ax.set_axisbelow(True)
            ax.grid(b=True, which='major')
        if legend and self._num_base > 1:
            ax.legend()

        if self.dim != 3:
            ax.set_aspect("equal", "box")
            fig.tight_layout()

        if show:
            plt.show()
        return ax

    def __repr__(self) -> str:
        shape = str(self.shape) if self.shape else "None"
        return f"{self.__class__.__name__}(dim: {self.dim}, " \
               f"num_base: {self.num_base}, shape: {shape})"
