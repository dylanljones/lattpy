# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Basis object for defining the coordinate system and unit cell of a lattice."""

import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from typing import Union, Sequence, Callable
from .spatial import cell_size, cell_volume, WignerSeitzCell
from .plotting import subplot, draw_unit_cell

logger = logging.getLogger(__name__)

vecs_t = Union[float, Sequence[float], Sequence[Sequence[float]]]
basis_t = Union[float, Sequence[float], Sequence[Sequence[float]], "LatticeBasis"]

__all__ = ["basis_t", "LatticeBasis"]


class LatticeBasis:
    """Lattice basis for representing the coordinate system and unit cell of a lattice.

    The ``LatticeBasis`` object is the core of any lattice model. It defines the
    basis vectors and subsequently the coordinate system of the lattice and provides
    the necessary basis transformations between the world and lattice coordinate system.

    Attributes
    ----------
    dim
    vectors
    vectors3d
    norms
    cell_size
    cell_volume

    Methods
    -------
    chain
    square
    rectangular
    oblique
    hexagonal
    hexagonal3d
    sc
    fcc
    bcc
    itransform
    transform
    itranslate
    translate
    is_reciprocal
    reciprocal_vectors
    reciprocal_lattice
    get_neighbor_cells
    wigner_seitz_cell
    brillouin_zone
    plot_basis

    Parameters
    ----------
    basis: array_like or float or LatticeBasis
        The primitive basis vectors that define the unit cell of the lattice. If a
        ``LatticeBasis`` instance is passed it is copied and used as the new basis.
    **kwargs
        Key-word arguments. Used only when subclassing ``LatticeBasis``.

    Examples
    --------
    >>> import lattpy as lp
    >>> import matplotlib.pyplot as plt
    >>> basis = lp.LatticeBasis.square()
    >>> _ = basis.plot_basis()
    >>> plt.show()

    """

    # Tolerance for reciprocal vectors/lattice
    RVEC_TOLERANCE: float = 1e-6

    # noinspection PyUnusedLocal
    def __init__(self, basis: basis_t, **kwargs):
        if isinstance(basis, LatticeBasis):
            basis = basis.vectors
        # Vector basis
        self._vectors = np.atleast_2d(basis).T
        self._vectors_inv = np.linalg.inv(self._vectors)
        self._dim = len(self._vectors)
        self._cell_size = cell_size(self.vectors)
        self._cell_volume = cell_volume(self.vectors)

    @classmethod
    def chain(cls, a: float = 1.0, **kwargs):
        """Initializes a one-dimensional lattice."""
        return cls(a, **kwargs)

    @classmethod
    def square(cls, a: float = 1.0, **kwargs):
        """Initializes a 2D lattice with square basis vectors."""
        return cls(a * np.eye(2), **kwargs)

    @classmethod
    def rectangular(cls, a1: float = 1.0, a2: float = 1.0, **kwargs):
        """Initializes a 2D lattice with rectangular basis vectors."""
        return cls(np.array([[a1, 0], [0, a2]]), **kwargs)

    @classmethod
    def oblique(cls, alpha: float, a1: float = 1.0, a2: float = 1.0, **kwargs):
        """Initializes a 2D lattice with oblique basis vectors."""
        vectors = np.array([[a1, 0], [a2 * np.cos(alpha), a2 * np.sin(alpha)]])
        return cls(vectors, **kwargs)

    @classmethod
    def hexagonal(cls, a: float = 1.0, **kwargs):
        """Initializes a 2D lattice with hexagonal basis vectors."""
        vectors = a / 2 * np.array([[3, np.sqrt(3)], [3, -np.sqrt(3)]])
        return cls(vectors, **kwargs)

    @classmethod
    def hexagonal3d(cls, a: float = 1.0, az: float = 1.0, **kwargs):
        """Initializes a 3D lattice with hexagonal basis vectors."""
        vectors = (
            a / 2 * np.array([[3, np.sqrt(3), 0], [3, -np.sqrt(3), 0], [0, 0, az]])
        )
        return cls(vectors, **kwargs)

    @classmethod
    def sc(cls, a: float = 1.0, **kwargs):
        """Initializes a 3D simple cubic lattice."""
        return cls(a * np.eye(3), **kwargs)

    @classmethod
    def fcc(cls, a: float = 1.0, **kwargs):
        """Initializes a 3D face centered cubic lattice."""
        vectors = a / 2 * np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        return cls(vectors, **kwargs)

    @classmethod
    def bcc(cls, a: float = 1.0, **kwargs):
        """Initializes a 3D body centered cubic lattice."""
        vectors = a / 2 * np.array([[1, 1, 1], [1, -1, 1], [-1, 1, 1]])
        return cls(vectors, **kwargs)

    # ==================================================================================

    @property
    def dim(self) -> int:
        """int: The dimension of the lattice."""
        return self._dim

    @property
    def vectors(self) -> np.ndarray:
        """np.ndarray: Array containing the basis vectors as rows."""
        return self._vectors.T

    @property
    def vectors3d(self) -> np.ndarray:
        """np.ndarray: The basis vectors expanded to three dimensions."""
        vectors = np.eye(3)
        vectors[: self.dim, : self.dim] = self._vectors
        return vectors.T

    @property
    def norms(self) -> np.ndarray:
        """np.ndarray: Lengths of the basis vectors."""
        return np.linalg.norm(self._vectors, axis=0)

    @property
    def cell_size(self) -> np.ndarray:
        """np.ndarray: The shape of the box spawned by the basis vectors."""
        return self._cell_size

    @property
    def cell_volume(self) -> float:
        """float: The volume of the unit cell defined by the basis vectors."""
        return self._cell_volume

    def itransform(
        self, world_coords: Union[Sequence[int], Sequence[Sequence[int]]]
    ) -> np.ndarray:
        """Transform the world coords ``(x, y, ...)`` into the basis coords
        ``(n, m, ...)``.

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

        >>> latt = LatticeBasis([[2, 0], [0, 1]])

        Transform points into the coordinate system of the lattice:

        >>> latt.itransform([2, 0])
        [1. 0.]

        >>> latt.itransform([4, 0])
        [2. 0.]

        >>> latt.itransform([0, 1])
        [0. 1.]
        """
        world_coords = np.atleast_1d(world_coords)
        return np.inner(world_coords, self._vectors_inv)

    def transform(
        self, basis_coords: Union[Sequence[int], Sequence[Sequence[int]]]
    ) -> np.ndarray:
        """Transform the basis-coords ``(n, m, ...)`` into the world coords
        ``(x, y, ...)``.

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
        :math:`a_2 = (0, 1)` :

        >>> latt = LatticeBasis([[2, 0], [0, 1]])

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

    def translate(
        self,
        nvec: Union[int, Sequence[int], Sequence[Sequence[int]]],
        r: Union[float, Sequence[float]] = 0.0,
    ) -> np.ndarray:
        r"""Translates the given postion vector ``r`` by the translation vector ``n``.

        The position is calculated using the translation vector :math:`n` and the
        atom position in the unitcell :math:`r`:

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
        r_trans : (..., N) np.ndarray
            The translated position.

        Examples
        --------
        Construct a lattice with basis vectors :math:`a_1 = (2, 0)` and
        :math:`a_2 = (0, 1)`:

        >>> latt = LatticeBasis([[2, 0], [0, 1]])

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
        x : (..., N) array_like or float
            Position vector in cartesian coordinates.

        Returns
        -------
        nvec : (..., N) np.ndarray
            Translation vector in the lattice basis.
        r : (..., N) np.ndarray, optional
            The position in real-space.

        Examples
        --------
        Construct a lattice with basis vectors :math:`a_1 = (2, 0)` and
        :math:`a_2 = (0, 1)`:

        >>> latt = LatticeBasis([[2, 0], [0, 1]])
        >>> latt.itranslate([2, 0])
        (array([1., 0.]), array([0., 0.]))

        >>> latt.itranslate([2.5, 0.5])
        (array([1., 0.]), array([0.5, 0.5]))
        """
        x = np.atleast_1d(x)
        itrans = self.itransform(x)
        nvec = np.floor(itrans).astype(np.int64)
        r = x - self.translate(nvec)
        return nvec, r

    def is_reciprocal(self, vecs: vecs_t, tol: float = RVEC_TOLERANCE) -> bool:
        r"""Checks if the given vectors are reciprocal to the lattice vectors.

        The lattice- and reciprocal vectors :math:`a_i` and :math:`b_i` must satisfy
        the relation

        .. math::
            a_i \cdot b_i = 2 \pi \delta_{ij}

        To check the given vectors, the difference of each dot-product is compared to
        :math:`2\pi` with the given tolerance.

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

    def reciprocal_vectors(
        self, tol: float = RVEC_TOLERANCE, check: bool = False
    ) -> np.ndarray:
        r"""Computes the reciprocal basis vectors of the bravais lattice.

        The lattice- and reciprocal vectors :math:`a_i` and :math:`b_i` must satisfy
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

        >>> latt = LatticeBasis(np.eye(2))
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
        rvecs = rvecs[: self.dim, : self.dim]

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
    def reciprocal_lattice(self, min_negative: bool = False):
        """Creates the lattice in reciprocal space.

        Parameters
        ----------
        min_negative : bool, optional
            If True the reciprocal vectors are scaled such that
            there are fewer negative elements than positive ones.

        Returns
        -------
        rlatt : LatticeBasis
            The lattice in reciprocal space

        See Also
        --------
        reciprocal_vectors : Constructs the reciprocal vectors used for the
            reciprocal lattice

        Examples
        --------
        Reciprocal lattice of the square lattice:

        >>> latt = LatticeBasis(np.eye(2))
        >>> rlatt = latt.reciprocal_lattice()
        >>> rlatt.vectors
        [[6.28318531 0.        ]
         [0.         6.28318531]]
        """
        rvecs = self.reciprocal_vectors(min_negative)
        rlatt = self.__class__(rvecs)
        return rlatt

    def get_neighbor_cells(
        self,
        distidx: int = 0,
        include_origin: bool = True,
        comparison: Callable = np.isclose,
    ) -> np.ndarray:
        """Find all neighboring unit cells of the unit cell at the origin.

        Parameters
        ----------
        distidx : int, default
            Index of distance to neighboring cells, default is 0 (nearest neighbors).
        include_origin : bool, optional
            If True the origin is included in the set.
        comparison : callable, optional
            The method used for comparing distances.

        Returns
        -------
        indices : np.ndarray
            The lattice indeices of the neighboring unit cells.

        Examples
        --------
        >>> latt = LatticeBasis(np.eye(2))
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
            If True the reciprocal vectors are scaled such that
            there are fewer negative elements than positive ones.

        Returns
        -------
        ws_cell : WignerSeitzCell
            The Wigner-Seitz cell of the reciprocal lattice.
        """
        rvecs = self.reciprocal_vectors(min_negative)
        rlatt = self.__class__(rvecs)
        return rlatt.wigner_seitz_cell()

    def plot_basis(
        self,
        lw: float = None,
        ls: str = "--",
        margins: Union[Sequence[float], float] = 0.1,
        grid: bool = False,
        show_cell: bool = True,
        show_vecs: bool = True,
        adjustable: str = "box",
        ax: Union[plt.Axes, Axes3D] = None,
        show: bool = False,
    ) -> Union[plt.Axes, Axes3D]:  # pragma: no cover
        """Plot the lattice basis.

        Parameters
        ----------
        lw : float, optional
            The line width used for plotting the unit cell outlines.
        ls : str, optional
            The line style used for plotting the unit cell outlines.
        margins : Sequence[float] or float, optional
            The margins of the plot.
        grid : bool, optional
            If True, draw a grid in the plot.
        show_vecs : bool, optional
            If True the first unit-cell is drawn.
        show_cell : bool, optional
            If True the outlines of the unit cell are plotted.
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
            draw_unit_cell(ax, vectors, show_cell, lw=lw, color="k", ls=ls, zorder=hopz)
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
        return f"{self.__class__.__name__}(dim: {self.dim})"
