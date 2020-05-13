# coding: utf-8
"""
Created on 08 Apr 2020
author: Dylan Jones
"""
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from .core.vector import VectorBasis, vrange, distance
from .core.atoms import Atom
from .core.errors import ConfigurationError
from .core.plotting import draw_sites, draw_cell


class BravaisLattice(VectorBasis):

    DIST_DECIMALS = 5
    REC_TOLERANCE = 1e-5
    MIN_DISTS = 3

    def __init__(self, vectors):
        super().__init__(vectors)
        self.origin = np.zeros(self.dim)

        # Atom data
        self.n_base = 0
        self.atoms = list()
        self.atom_positions = list()
        self.n_dist = 0
        self.distances = list()
        self._base_neighbors = list()

    @classmethod
    def chain(cls, a=1.0):
        return cls(a)

    @classmethod
    def square(cls, a=1.0):
        return cls(a * np.eye(2))

    @classmethod
    def rectangular(cls, a1=1.0, a2=1.0):
        return cls(np.array([[a1, 0], [0, a2]]))

    @classmethod
    def hexagonal(cls, a=1.0):
        vectors = a/2 * np.array([[3, np.sqrt(3)], [3, -np.sqrt(3)]])
        return cls(vectors)

    @classmethod
    def sc(cls, a=1.0):
        return cls(a * np.eye(3))

    @classmethod
    def fcc(cls, a=1.0):
        vectors = a/2 * np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        return cls(vectors)

    @classmethod
    def bcc(cls, a=1.0):
        vectors = a/2 * np.array([[1, 1, 1], [1, -1, 1], [-1, 1, 1]])
        return cls(vectors)

    def copy(self):
        """ Creates a (deep) copy of the lattice instance"""
        latt = self.__class__(self._vectors.copy().T)
        if self.n_base:
            latt.n_base = self.n_base
            latt.atoms = self.atoms.copy()
            latt.atom_positions = self.atom_positions.copy()
            latt.distances = self.distances.copy()
            latt._base_neighbors = self._base_neighbors.copy()
        return latt

    def save(self, file='lattice.pkl'):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file='lattice.pkl'):
        with open(file, "rb") as f:
            latt = pickle.load(f)
        return latt

    # =========================================================================

    def is_reciprocal(self, vecs, tol=REC_TOLERANCE):
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

    def reciprocal_vectors(self, tol=REC_TOLERANCE, check=False):
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

    def reciprocal_lattice(self, min_negative=False):
        """ Creates the lattice in reciprocal space

        Parameters
        ----------
        min_negative: bool, optional
            If 'True' the reciprocal vectors are scales such that the
            there are fewer negative elements than positive ones.

        Returns
        -------
        rlatt: object
        """
        rvecs = self.reciprocal_vectors(min_negative)
        rlatt = self.__class__(rvecs)
        if self.n_base:
            rlatt.n_base = self.n_base
            rlatt.atoms = self.atoms.copy()
            rlatt.atom_positions = self.atom_positions.copy()
            rlatt.calculate_distances(self.n_dist)
        return rlatt

    def translate(self, nvec, r=0):
        """ Translates the given postion vector r by the translation vector n.

        Parameters
        ----------
        nvec: (N) array_like
            Translation vector in the lattice basis.
        r: (N) array_like, optional
            The position in real-space. If no vector is passed only the translation is returned.

        Returns
        -------
        r_trans: (N) array_like
        """
        return r + (self._vectors @ nvec)

    def itranslate(self, v):
        """ Returns the lattice index and cell position leading to the given position in real space.

        Parameters
        ----------
        v: (N) array_like
            Position vector in real-space.

        Returns
        -------
        nvec: (N) array_like
            Translation vector in the lattice basis.
        r: (N) array_like, optional
            The position in real-space.
        """
        itrans = self._vectors_inv @ v
        nvec = np.floor(itrans)
        r = v - self.translate(nvec)
        return nvec, r

    def estimate_index(self, pos):
        """ Returns the nearest matching lattice index (n, alpha) for global position.

        Parameters
        ----------
        pos: array_like
            global site position.

        Returns
        -------
        n: np.ndarray
            estimated translation vector n
        """
        pos = np.asarray(pos)
        n = np.asarray(np.round(self._vectors_inv @ pos, decimals=0), dtype="int")
        return n

    def get_index(self, n=None, alpha=0):
        """ Returns lattice index in form of [n_1, ..., n_d, alpha]

        Parameters
        ----------
        n: (N) array_like or int
            translation vector.
        alpha: int, optional
            site index, default is 0.
        Returns
        -------
        index: (N) np.ndarray
        """
        if n is None:
            n = np.zeros(self.dim)
        return np.append(n, alpha)

    def get_position(self, n=None, alpha=0):
        """ Returns the position for a given translation vector and site index

        Parameters
        ----------
        n: (N) array_like or int
            translation vector.
        alpha: int, optional
            site index, default is 0.
        Returns
        -------
        pos: (N) np.ndarray
        """
        r = self.atom_positions[alpha]
        if n is None:
            return r
        n = np.atleast_1d(n)
        return r + (self._vectors @ n)  # self.translate(n, r)

    def get_atom(self, alpha):
        """ Returns the Atom object of the given atom in the unit cell

        Parameters
        ----------
        alpha: int
            Index of the atom in the unit cell.

        Returns
        -------
        atom: Atom
        """
        return self.atoms[alpha]

    def get_atom_attrib(self, alpha, attrib, default=None):
        """ Returns an attribute of a specific atom in the unit cell.

        Parameters
        ----------
        alpha: int
            Index of the atom in the unit cell.
        attrib: str
            Name of the atom attribute.
        default: str or int or float or object, optional
            Default value used if the attribute couln't be found in the Atom dictionary.

        Returns
        -------
        attrib: str or int or float or object
        """
        atom = self.atoms[alpha]
        return atom.attrib(attrib, default)

    def translate_cell(self, n):
        """ Translates all sites of the unit cell

        Parameters
        ----------
        n: np.ndarray
            translation vector.

        Yields
        -------
        pos: np.ndarray
            positions of the sites in the translated unit cell
        """
        n = np.atleast_1d(n)
        for alpha in range(self.n_base):
            yield self.get_position(n, alpha)

    def distance(self, idx0, idx1, decimals=DIST_DECIMALS):
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

    def _neighbour_range(self, n=None, cell_range=1):
        """ Get all neighbouring translation vectors of a given cell position

        Parameters
        ----------
        n: array_like, optional
            translation vector of unit cell, the default is the origin.
        cell_range: int, optional
            Range of neighbours, the default is 1.

        Returns
        -------
        trans_vectors: list
        """
        n = np.zeros(self.dim) if n is None else n
        offset = cell_range + 2
        ranges = [np.arange(n[d] - offset, n[d] + offset + 1) for d in range(self.dim)]
        n_vecs = vrange(ranges)
        for n in n_vecs:
            for alpha in range(self.n_base):
                yield (n, alpha)

    def calculate_neighbours(self, n=None, alpha=0, distidx=0, array=False):
        """ Find all neighbours of given site and return the lattice indices.

        Parameters
        ----------
        n: array_like, optional
            translation vector of site, the default is the origin.
        alpha: int, optional
            site index, default is 0.
        distidx: int, default
            index of distance to neighbours, defauzlt is 0 (nearest neighbours).
        array: bool, optional
            if true, return lattice index (n, alpha) as single array.
            The default is False.

        Returns
        -------
        indices: list
        """
        n = np.zeros(self.dim) if n is None else n
        idx = n, alpha
        dist = self.distances[distidx]
        indices = list()
        for idx1 in self._neighbour_range(n, distidx):
            # if np.isclose(self.distance(idx, idx1), dist, atol=1e-5):
            if np.round(abs(self.distance(idx, idx1) - dist), decimals=self.DIST_DECIMALS) == 0.0:
                if array:
                    idx1 = [*idx1[0], idx1[1]]
                indices.append(idx1)
        return indices

    def calculate_distances(self, num_dist=1):
        """ Calculates the ´n´ lowest distances between sites in the lattice and the neighbours of the cell.

        Checks distances between all sites of the bravais lattice and saves n lowest values.
        The neighbor lattice-indices of the unit-cell are also stored for later use.
        This speeds up many calculations like finding nearest neighbours.

        Raises
        ------
        ConfigurationError
            Raised if no atoms where added to the lattice.
            The atoms in the unit cell are needed for computing the distances in the lattice.

        Parameters
        ----------
        num_dist: int, optional
            Number of distances of lattice structure to calculate. If 'None' the number of atoms is used.
            The default is 1 (nearest neighbours).
        """
        if len(self.atoms) == 0:
            raise ConfigurationError("No atoms found in the lattice!", "Use 'add_atom' to add an 'Atom'-object")
        if num_dist is None:
            num_dist = len(self.atoms)
        # Calculate n lowest distances of lattice structure
        n = max(num_dist, self.MIN_DISTS) + 1
        n_vecs = vrange(self.dim * [np.arange(-n, n)])
        r_vecs = list()
        for nvec in n_vecs:
            for alpha in range(self.n_base):
                r_vecs.append(self.get_position(nvec, alpha))
        pairs = list(itertools.product(r_vecs, self.atom_positions))
        distances = list(set([distance(r1, r2, self.DIST_DECIMALS) for r1, r2 in pairs]))
        distances.sort()
        distances.remove(0.0)
        self.distances = distances[0:n - 1]
        self.n_dist = num_dist

        # Calculate cell-neighbors.
        neighbours = list()
        for alpha in range(self.n_base):
            site_neighbours = list()
            for i_dist in range(len(self.distances)):
                # Get neighbour indices of site for distance level
                site_neighbours.append(self.calculate_neighbours(alpha=alpha, distidx=i_dist, array=True))
            neighbours.append(site_neighbours)
        self._base_neighbors = neighbours

    def add_atom(self, pos=None, atom=None, neighbours=0, **kwargs):
        """ Adds a site to the basis of the lattice unit-cell.

        Parameters
        ----------
        pos: (N) array_like, optional
            Position of site in the unit-cell. The default is the origin of the cell.
            The size of the array has to match the dimension of the lattice.
        atom: str or Atom
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
            pos = np.asarray(pos)
        if any(np.all(pos == x) for x in self.atom_positions):
            raise ValueError(f"Position {pos} already occupied!")
        if isinstance(atom, Atom):
            atom = atom
        else:
            atom = Atom(atom, **kwargs)
        self.atoms.append(atom)
        self.atom_positions.append(np.asarray(pos))
        self.n_base = len(self.atom_positions)
        if neighbours:
            self.calculate_distances(neighbours)
        return atom

    def get_neighbours(self, n=None, alpha=0, distidx=0):
        """ Returns the neighours of a given site by transforming stored neighbour indices.

        Raises
        ------
        ConfigurationError
            Raised if the lattice distances haven't been computed.

        Parameters
        ----------
        n: array_like, optional
            translation vector of site, the default is the origin.
        alpha: int, optional
            site index, default is 0.
        distidx: int, default
            index of distance to neighbours, defauzlt is 0 (nearest neighbours).

        Returns
        -------
        indices: np.ndarray
        """
        if n is None:
            n = np.zeros(self.dim)
        if not self._base_neighbors:
            hint = "Use the 'neighbours' keyword of 'add_atom' or call 'calculate_distances' after adding the atoms!"
            raise ConfigurationError("Base neighbours not configured.", hint)
        n = np.atleast_1d(n)
        transformed = list()
        for idx in self._base_neighbors[alpha][distidx]:
            idx_t = idx.copy()
            idx_t[:-1] += n
            transformed.append(idx_t)
        return transformed

    def get_neighbour_vectors(self, alpha=0, distidx=0, include_zero=False):
        """ Returns the neighours of a given site by transforming stored neighbour indices.

        Raises
        ------
        ConfigurationError
            Raised if the lattice distances haven't been computed.

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
        vectors: np.ndarray
        """
        if not self._base_neighbors:
            hint = "Use the 'neighbours' keyword of 'add_atom' or call 'calculate_distances' after adding the atoms!"
            raise ConfigurationError("Base neighbours not configured.", hint)
        pos0 = self.atom_positions[alpha]
        vectors = list()
        if include_zero:
            vectors.append(np.zeros(self.dim))
        for idx in self._base_neighbors[alpha][distidx]:
            pos1 = self.get_position(idx[:-1], idx[-1])
            vectors.append(pos1 - pos0)
        return vectors

    def get_neighbour_pairs(self, distidx=0):
        for alpha1 in range(self.n_base):
            pos0 = self.get_position(alpha=alpha1)
            for idx in self.get_neighbours(alpha=alpha1, distidx=distidx):
                n, alpha2 = idx[:-1], idx[-1]
                pos1 = self.get_position(n, alpha2)
                delta = pos1 - pos0
                yield delta, alpha1, alpha2

    def get_base_atom_dict(self, atleast2d=True):
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
        for atom, pos in zip(self.atoms, self.atom_positions):
            if atleast2d and self.dim == 1:
                pos = np.array([pos, 0])

            if atom.name in atom_pos.keys():
                atom_pos[atom].append(pos)
            else:
                atom_pos[atom] = [pos]
        return atom_pos

    # =========================================================================

    def plot_cell(self, show=True, ax=None, color='k', lw=2, legend=True, margins=0.25,
                  show_atoms=True, outlines=True, grid=True):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d" if self.dim == 3 else None)
        else:
            fig = ax.get_figure()

        # Plot atoms in the unit cell
        if show_atoms and self.n_base:
            atom_pos = self.get_base_atom_dict()
            for atom, positions in atom_pos.items():
                draw_sites(ax, positions, size=atom.size, color=atom.col, label=atom.label())

        # Draw unit vectors and the cell they spawn.
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
        if legend and self.n_base:
            ax.legend()

        fig.tight_layout()
        if show:
            plt.show()
        return ax
