# coding: utf-8
"""
Created on 08 Apr 2020
author: Dylan Jones
"""
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from .core import Atom, LatticeCache
from .core.vector import VectorBasis, vrange, distance
from .core.plotting import draw_cell, draw_sites, draw_lines, draw_indices, set_padding


class ConfigurationError(Exception):

    def __init__(self, msg='', hint=''):
        if hint:
            msg += f'({hint})'
        super().__init__(msg)


class Lattice(VectorBasis):

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

        # Lattice Cache
        self.data = LatticeCache()
        self.shape = None
        self.periodic_axes = list()

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
        if self.data:
            latt.shape = self.shape.copy()
            latt.data = self.data.copy()
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

    @property
    def n_sites(self):
        """ int: Number of sites in cached lattice data"""
        return self.data.n

    def alpha(self, i):
        """ Returns the atom component of the lattice index of the given site

        Parameters
        ----------
        i: int
            Site index in the cached lattice data.

        Returns
        -------
        alpha: int
        """
        return self.data.indices[i][-1]

    def position(self, i):
        """ Returns the position for a given site in the cached lattice data.

        Parameters
        ----------
        i: int
            Site index in the cached lattice data.

        Returns
        -------
        pos: (N) np.ndarray
        """
        return self.get_position(*self.data.get_index(i))

    def neighbours(self, i, distidx=0, unique=False):
        """ Returns the neighours of a given site in the cached lattice data.

        Parameters
        ----------
        i: int
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
        for didx in distidx:
            neighbours += self.data.get_neighbours(i, didx)
        if unique:
            neighbours = [idx for idx in neighbours if idx > i]
        return neighbours

    def nearest_neighbours(self, i, unique=False):
        """ Returns the nearest neighours of a given site in the cached lattice data.

        Parameters
        ----------
        i: int
            Site index in the cached lattice data.
        unique: bool, optional
            If 'True', each unique pair is only return once.

        Returns
        -------
        indices: list of int
        """
        return self.neighbours(i, 0, unique)

    def iter_neighbours(self, i, unique=False):
        """ Iterate over the neighbours of all distances of a given site in the cached lattice data.

        Parameters
        ----------
        i: int
            Site index in the cached lattice data.
        unique: bool, optional
            If 'True', each unique pair is only return once.

        Yields
        -------
        distidx: int
        siteidx: int
        """
        for distidx in range(self.n_dist):
            for j in self.neighbours(i, distidx, unique):
                yield distidx, j

    def _build_indices_inbound(self, shape, pos=None):
        shape = np.asarray(shape)
        if pos is None:
            pos = self.origin
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
            for alpha in range(self.n_base):
                # Check if index is in the real space shape
                include = True
                r = self.get_position(n, alpha)
                pos = self.origin if pos is None else pos
                for d in range(self.dim):
                    if not pos[d] <= r[d] <= pos[d] + shape[d]:
                        include = False
                        break
                if include:
                    indices.append([*n, alpha])
        return indices

    def _build_indices(self, shape, pos=None):
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
            for alpha in range(self.n_base):
                indices.append([*n, alpha])
        return indices

    def _construct(self, new_indices, new_neighbours=None, site_indices=None, window=None):
        """ Constructs the index array and computes the neighbour indices.

        Parameters
        ----------
        new_indices: array_like
            Array of the new indices in the form of .math:'[n_1, .., n_N, alpha]' to add to the lattice.
            If the lattice doesn't have data yet a new array is created.
        new_neighbours: array_like, optional
            Optional array of new neighbours to add. by default a new array is created.
            This is used for adding new connections to an extisting lattice block.
        site_indices: array_like, optional
            Optional indices to calculate neighbours. This can be used for computing only
            neighbours in the region of a connection.
        window: int, optional
            Window for looking up neighbours. This can speed up the computation significally.
            Generally at least a few layers of the lattice should be searched. By default the whole
            range of the lattice sites is used.

        Returns
        -------
        indices: array_like
        neighbours: array_like
        """
        num_sites = len(new_indices)
        n_dist = self.n_dist
        if not n_dist:
            hint = "Use the 'neighbours' keyword of 'add_atom' or call 'calculate_distances' after adding the atoms!"
            raise ConfigurationError("Base neighbours not configured.", hint)

        # Initialize new indices and the empty neighbour array
        new_indices = np.array(new_indices)
        if new_neighbours is None:
            new_neighbours = [[set() for _ in range(n_dist)] for _ in range(num_sites)]

        # get all sites and neighbours (cached and new)
        if self.data:
            all_indices = np.append(self.data.indices, new_indices, axis=0)
            all_neighbours = self.data.neighbours + new_neighbours
        else:
            all_indices = new_indices
            all_neighbours = new_neighbours

        # Find neighbours of each site in the "new_indices" list and store the neighbours
        if window is None:
            window = len(all_indices)
        offset = self.data.n
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

            # Get neighbour indices of site
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

        return all_indices, all_neighbours

    def set_data(self, indices, neighbours):
        """ Sets cached data and recomputes real-space shape of lattice

        Parameters
        ----------
        indices: array_like
            Lattice indices that will be saved.
        neighbours: array_like
            Lattice site neighbours that will be saved.
        """
        # Set data and recompute real-space shape of lattice
        self.data.set(indices, neighbours)
        points = [self.position(i) for i in range(self.data.n)]
        limits = np.array([np.min(points, axis=0), np.max(points, axis=0)])
        self.shape = limits[1] - limits[0]

    def build(self, shape, inbound=True, pos=None):
        """ Constructs the indices and neighbours of a new finite size lattice and stores the data

        Parameters
        ----------
        shape: array_like
            shape of finite size lattice to build.
        inbound: bool, optional
            If 'True' the shape will be interpreted in real-space. Only lattice-sites in this shape
            will be added to the data. This ensures nicer shapes of the lattice. Otherwise the shape is
            constructed in the basis if the unit-vectors. The default is 'True'
        pos: array_like, optional
            Optional position of the section to build. If 'None' the origin is used.
        """
        self.data.reset()
        shape = np.atleast_1d(shape)
        if len(shape) != self.dim:
            raise ValueError(f"Dimension of shape {len(shape)} doesn't match the dimension of the lattice {self.dim}")

        # Compute indices and initialize neighbour array
        if inbound:
            indices = self._build_indices_inbound(shape, pos=pos)
        else:
            indices = self._build_indices(shape.astype('int'), pos=pos)

        # Compute neighbours of indices
        if len(indices) > 100:
            window = int(len(indices) * 0.5)
        else:
            window = None
        indices, neighbours = self._construct(indices, window=window)
        self.set_data(indices, neighbours)

    def build_centered(self, shape):
        """ Builds a centered lattice in the given (real world) coordinates.

        Parameters
        ----------
        shape: array_like
            shape of finite size lattice to build.
        """
        center = np.asarray(shape) / 2
        self.build(shape, inbound=True, pos=-center)

    def add_x(self, latt, shift=True):
        n_new = latt.n_sites
        new_data = latt.data.copy()
        new_indices = new_data.indices
        new_neighbours = list()
        if shift:
            new_indices[:, 0] += self.estimate_index((self.shape[0], 0))[0] + 1
        for site_neighbours in new_data.neighbours:
            shifted = list()
            for dist_neighbours in site_neighbours:
                shifted.append(set([x + self.n_sites for x in dist_neighbours]))
            new_neighbours.append(shifted)

        # Find neighbours of connecting section
        window = range(0, int(n_new / 2))
        indices, neighbours = self._construct(new_indices, new_neighbours, site_indices=window)

        # Set data and recompute real-space shape of lattice
        self.data.set(indices, neighbours)
        points = [self.position(i) for i in range(self.data.n)]
        limits = np.array([np.min(points, axis=0), np.max(points, axis=0)])
        self.shape = limits[1] - limits[0]

    def __add__(self, other):
        new = self.copy()
        new.add_x(other)
        return new

    def set_periodic(self, axis=0):
        """ Sets periodic boundary conditions alogn the given axis.

        Adds the indices of the neighbours cycled around the given axis.

        Notes
        -----
        The lattice has to be built before applying the periodic boundarie conditions.
        Also the lattice has to be at least three atoms big in the specified directions.

        Parameters
        ----------
        axis: int or (N) array_like, optional
            One or multiple axises to apply the periodic boundary conditions.
            The default is the x-direction. If the axis is `None` the perodic boundary
            conditions will be removed.
        """
        if axis is None:
            self.data.set_periodic_neighbours(None)
            return
        axis = np.atleast_1d(axis)
        self.periodic_axes = axis
        n = self.n_sites
        neighbours = [[set() for _ in range(self.n_dist)] for _ in range(self.n_sites)]
        if self.dim == 1:
            for distidx in range(self.n_dist):
                i = distidx
                j = n - distidx - 1
                neighbours[i][distidx].add(j)
                neighbours[j][distidx].add(i)
        else:
            for ax in axis:
                offset = np.zeros(self.dim, dtype="float")
                offset[ax] = self.shape[ax] + 0.1 * self.cell_size[ax]
                pos = offset

                nvec = pos @ np.linalg.inv(self._vectors.T)
                nvec[ax] = np.ceil(nvec[ax])
                nvec = np.round(nvec, decimals=0).astype("int")
                for i in range(n):
                    pos1 = self.position(i)
                    for j in range(0, n):
                        pos2 = self.translate(nvec, self.position(j))
                        dist = distance(pos1, pos2, self.DIST_DECIMALS)
                        if dist in self.distances:
                            i_dist = self.distances.index(dist)
                            if i_dist < self.n_dist:
                                neighbours[i][i_dist].add(j)
                                neighbours[j][i_dist].add(i)
        self.data.set_periodic_neighbours(neighbours)

    def transform_periodic(self, pos, ax, cell_offset=0.0):
        """ Transforms the given position along the given axis by the shape of the lattice.

        Parameters
        ----------
        pos: array_like
            Position coordinates to tranform.
        ax: int:
            The axis that the position will be transformed along.
        cell_offset: float, optional
            Optional offset in units of the cell size.

        Returns
        -------
        transformed: np.ndarray
        """
        delta = np.zeros(self.dim, dtype="float")
        # Get cell offset along axis
        delta[ax] = self.cell_size[ax] * cell_offset
        # Get translation along axis
        delta[ax] += self.shape[ax]
        # Get transformation direction (to nearest periodic cell)
        sign = -1 if pos[ax] > self.shape[ax] / 2 else +1
        return pos + sign * delta

    def atom_positions_dict(self, indices=None, atleast2d=True):
        """ Returns a dictionary containing the positions for each type of the atoms.

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
        indices = self.data.indices if indices is None else indices
        atom_pos = dict()
        for idx in indices:
            n, alpha = idx[:-1], idx[-1]
            atom = self.atoms[alpha]
            pos = self.get_position(n, alpha)
            if atleast2d and self.dim == 1:
                pos = np.array([pos, 0])

            if atom.name in atom_pos.keys():
                atom_pos[atom].append(pos)
            else:
                atom_pos[atom] = [pos]
        return atom_pos

    def all_positions(self):
        """ Returns all positions, independent of the atom type, for the lattice.

        Returns
        -------
        positions: array_like
        """
        return np.asarray([self.position(i) for i in range(self.n_sites)])

    def get_connections(self, atleast2d=True):
        """ Returns all pairs of neighbours in the lattice

        Parameters
        ----------
        atleast2d: bool, optional
            If 'True', one-dimensional coordinates will be casted to 2D vectors.

        Returns
        -------
        connections: array_like
        """
        conns = list()
        for i in range(self.n_sites):
            neighbor_list = self.data.neighbours[i]
            for distidx in range(self.n_dist):
                for j in neighbor_list[distidx]:
                    if j > i:
                        p1 = self.position(i)
                        p2 = self.position(j)
                        if atleast2d and self.dim == 1:
                            p1 = np.array([p1, 0])
                            p2 = np.array([p2, 0])
                        conns.append([p1, p2])
        return np.asarray(conns)

    def get_periodic_segments(self, scale=1.0, atleast2d=True):
        """ Returns all pairs of peridoic neighbours in the lattice

        Parameters
        ----------
        scale: float, optional
        atleast2d: bool, optional
            If 'True', one-dimensional coordinates will be casted to 2D vectors.

        Returns
        -------
        connections: array_like
        """
        conns = list()
        for i in range(int(self.n_sites)):
            p1 = self.position(i)
            for distidx in range(self.n_dist):
                neighbours = self.data.periodic_neighbours[i][distidx]
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
                        p2 = p1 + scale * v
                        conns.append([p1, p2])
        return conns

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

    def plot(self, show=True, ax=None, lw=1., legend=True, grid=False,
             show_periodic=True, show_indices=False, show_cell=False):
        """ Plot the cached lattice

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
            periodic_segments = self.get_periodic_segments(scale=0.5)
            draw_lines(ax, periodic_segments, color="0.5", lw=lw)

        for atom, positions in atom_pos.items():
            draw_sites(ax, positions, size=atom.size, color=atom.col, label=atom.label())
        if show_indices:
            positions = [self.position(i) for i in range(self.n_sites)]
            draw_indices(ax, positions)

        if self.dim == 1:
            w = self.cell_size[0]
            ax.set_ylim(-w, +w)
        elif self.dim == 2:
            set_padding(ax, self.cell_size[0]/2, self.cell_size[1]/2)
        else:
            ax.margins(0.1, 0.1, 0.1)

        if self.dim != 3:
            ax.set_aspect("equal", "box")

        if grid:
            ax.set_axisbelow(True)
            ax.grid(b=True, which='major')
        if legend and self.n_base > 1:
            ax.legend()

        fig.tight_layout()
        if show:
            plt.show()
        return ax
