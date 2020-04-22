# coding: utf-8
"""
Created on 08 Apr 2020
author: Dylan Jones
"""
import numpy as np
import copy
from .utils import vrange, distance, ConfigurationError
from .plotting import LatticePlot
from .base import BravaisLattice


class LatticeData:

    def __init__(self):
        self.indices = None
        self.neighbours = None
        self.periodic_neighbours = None

    @property
    def n(self):
        return len(self.indices) if self.indices is not None else 0

    def copy(self):
        data = LatticeData()
        if self.indices is not None:
            indices = self.indices.copy()
            neighbours = copy.deepcopy(self.neighbours)
            data.set(indices, neighbours)
        if self.periodic_neighbours is not None:
            data.set_periodic_neighbours(copy.deepcopy(self.periodic_neighbours))
        return data

    def reset(self):
        self.indices = None
        self.neighbours = None
        self.periodic_neighbours = None

    def set(self, indices, neighbours):
        self.indices = np.asarray(indices)
        self.neighbours = neighbours

    def set_indices(self, indices):
        self.indices = np.asarray(indices)

    def set_neighbours(self, neighbours):
        self.neighbours = neighbours

    def set_periodic_neighbours(self, neighbours):
        self.periodic_neighbours = neighbours

    def get_nvec(self, i):
        return self.indices[i, :-1]

    def get_alpha(self, i):
        return self.indices[i, -1]

    def get_index(self, i):
        index = self.indices[i]
        return index[:-1], index[-1]

    def get_neighbours(self, i, dist=0):
        neighbours = list(self.neighbours[i][dist])
        if self.periodic_neighbours is not None:
            neighbours += list(self.periodic_neighbours[i][dist])
        return neighbours

    def __bool__(self):
        return self.indices is not None


class Lattice(BravaisLattice):

    def __init__(self, vectors):
        super().__init__(vectors)

        # Lattice Cache
        self.data = LatticeData()
        self.shape = None

    def copy(self):
        latt = super().copy()
        if self.data:
            latt.shape = self.shape.copy()
            latt.data = self.data.copy()
        return latt

    @property
    def n_sites(self):
        return self.data.n

    def alpha(self, i):
        return self.data.indices[i][-1]

    def position(self, i):
        return self.get_position(*self.data.get_index(i))

    def neighbours(self, i, distidx=0, unique=False):
        if not hasattr(distidx, '__len__'):
            distidx = [distidx]
        neighbours = list()
        for didx in distidx:
            neighbours += self.data.get_neighbours(i, didx)
        if unique:
            neighbours = [idx for idx in neighbours if idx > i]
        return neighbours

    def nearest_neighbours(self, i, unique=False):
        return self.neighbours(i, 0, unique)

    def iter_neighbours(self, i, unique=False):
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
            Window for looking for neighbours. This can speed up the computation significally.
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
                for idx in self.get_neighbours(site_idx, i_dist):
                    # Find site of neighbour and store if in cache
                    hop_idx = np.where(np.all(site_window == idx, axis=1))[0]
                    if len(hop_idx):
                        j_site = hop_idx[0] + i0
                        all_neighbours[i_site][i_dist].add(j_site)
                        all_neighbours[j_site][i_dist].add(i_site)
            # all_neighbours = self._set_neighbours(site, idx, all_indices, all_neighbours, window)

        return all_indices, all_neighbours

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

        # Set data and recompute real-space shape of lattice
        self.data.set(indices, neighbours)
        points = [self.position(i) for i in range(self.data.n)]
        limits = np.array([np.min(points, axis=0), np.max(points, axis=0)])
        self.shape = limits[1] - limits[0]

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
        n = self.n_sites
        neighbours = [[set() for _ in range(self.n_dist)] for _ in range(self.n_sites)]
        for ax in axis:
            offset = np.zeros(self.dim, dtype="float")
            offset[ax] = self.shape[ax] + 0.1 * self.cell_size[ax]
            pos = offset

            nvec = pos @ np.linalg.inv(self.vectors.T)
            nvec[ax] = np.ceil(nvec[ax])
            nvec = np.round(nvec, decimals=0).astype("int")

            for i in range(n):
                pos1 = self.position(i)
                for j in range(0, n):
                    pos2 = self.translate(nvec, self.position(j))
                    dist = np.round(distance(pos1, pos2), decimals=self.DIST_DECIMALS)
                    if dist in self.distances:
                        i_dist = self.distances.index(dist)
                        if i_dist < self.n_dist:
                            neighbours[i][i_dist].add(j)
                            neighbours[j][i_dist].add(i)
        self.data.set_periodic_neighbours(neighbours)

    # =========================================================================

    def plot(self, show=True, plot=None, legend=True, margins=0.1, padding=None, lw=1.,
             show_hop=True, show_indices=False, show_cell=False):
        """ Plot the cached lattice

        Parameters
        ----------
        plot: LatticePlot, optional
            Parent plot. If None, a new plot is initialized.
        show: bool, default: True
            parameter for pyplot
        legend: bool, optional
            Flag if legend is shown
        margins: float, default: None
            Relative padding used in the lattice plot. The default is '0.1'.
        padding: float, default: None
            Absolute padding used in the lattice plot. The default is 'None'.
        lw: float, default: 1
            Line width of the hopping connections
        show_hop: bool, default: True
            Draw hopping connections if True
        show_indices: bool, optional
            If 'True' the index of the sites will be shown.
        show_cell: bool, optional
            If 'True' the first unit-cell is drawn.

        Returns
        -------
        plot: LatticePlot
        """
        indices = self.data.indices
        neighbours = self.data.neighbours

        # Prepare site positions and hopping segments
        n_sites = len(indices)
        atom_pos = dict()
        positions = list()
        for idx in indices:
            n, alpha = idx[:-1], idx[-1]
            atom = self.atoms[alpha]
            pos = self.get_position(n, alpha)
            if self.dim == 1:
                pos = [pos, 0.0]

            if atom.name in atom_pos.keys():
                atom_pos[atom].append(pos)
            else:
                atom_pos[atom] = [pos]
            positions.append(pos)

        segments = list()
        if show_hop:
            for i in range(n_sites):
                neighbor_list = neighbours[i]
                for i_hop in range(self.n_dist):
                    for j in neighbor_list[i_hop]:
                        if j > i:
                            segments.append([positions[i], positions[j]])

        plot = plot or LatticePlot(dim3=self.dim == 3)
        plot.set_equal_aspect()
        for atom, positions in atom_pos.items():
            plot.draw_sites(atom, positions)
        plot.draw_linecollection(segments, color="k", lw=lw)
        if show_indices:
            positions = [self.position(i) for i in range(self.n_sites)]
            plot.print_indices(positions)

        if show_cell:
            self.plot_cell(plot=plot, show_atoms=False)
        if self.dim != 3:
            if padding is not None:
                plot.set_padding(padding)
            else:
                plot.set_margins(margins)

        if self.dim == 1 or self.shape[1] == 0:
            plot.set_limits(y=(-1, +1))

        plot.setup()
        if self.n_base > 1 or legend:
            plot.legend()

        plot.show(show)
        return plot
