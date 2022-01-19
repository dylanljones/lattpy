# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from typing import Union, Optional, Any, Iterator, Dict, Sequence, List, Tuple, Set
from .utils import SiteOccupiedError
from .spatial import distance, distances
from .unitcell import Atom

__all__ = ["Molecule"]


class Molecule:

    def __init__(self):
        self._atoms = list()
        self._positions = list()
        self._connections = list()

    @property
    def dim(self) -> int:
        """The dimension of the molecule positions."""
        return len(self._positions[0])

    @property
    def num_atoms(self) -> int:
        """Return the number of atoms in the molecule."""
        return len(self._atoms)

    @property
    def atoms(self) -> List[Atom]:
        """Return the atoms contained in the unitcell."""
        return self._atoms

    @property
    def positions(self) -> np.ndarray:
        """Return the positions of the atoms contained in the unitcell."""
        return np.asanyarray(self._positions)

    @property
    def positions3d(self) -> np.ndarray:
        """np.ndarray : Positions expanded to three dimensions."""
        positions = np.zeros((self.num_atoms, 3))
        positions[:, :self.dim] = self._positions
        return positions

    @property
    def connections(self) -> List[Set[int]]:
        return self._connections

    def add_atom(self, pos: Union[float, Sequence[float]] = None,
                 atom: Union[str, Dict[str, Any], Atom] = None, **kwargs) -> Atom:
        """ Adds a new atom to the molecule.

        Raises
        ------
        ValueError:
            A Value Error is raised if the position of the new atom is already occupied.

        Parameters
        ----------
        pos: (N) array_like or float, optional
            Position of site in the unit-cell. The default is the origin of the
            molecule. The size of the array has to match the dimension of the lattice.
        atom: str or dict or Atom, optional
            Identifier of the site. If a string is passed, a new instance
            is created.
        **kwargs
            Keyword arguments for ´Atom´ constructor. Only used if a new instance
            is created.

        Returns
        -------
        atom: Atom
            The added `Atom`-object
        """
        pos = np.atleast_1d(pos)
        if any(np.all(pos == x) for x in self._positions):
            raise SiteOccupiedError(atom, pos)

        if isinstance(atom, str):
            # Check if atom with the same name exists
            for at in self._atoms:
                if at.name == atom:
                    atom = at
                    break
            else:
                # Create new atom
                atom = Atom(atom, **kwargs)
        elif not isinstance(atom, Atom):
            atom = Atom(atom, **kwargs)

        self._atoms.append(atom)
        self._positions.append(np.asarray(pos))
        assert len(self._atoms) == len(self._positions)

        return atom

    def add_connection(self, atom1: int, atom2: int) -> None:
        if not self._connections:
            # Assumes all atoms are added
            self._positions = np.array(self._positions)
            self._connections = [set() for _ in range(self.num_atoms)]
        self._connections[atom1].add(atom2)
        self._connections[atom2].add(atom1)

    def copy(self) -> 'Molecule':
        """Creates a deep copy of the ``Molecule`` instance."""
        molecule = Molecule()
        for pos, atom in zip(self._positions, self._atoms):
            molecule.add_atom(pos.copy(), atom.copy())
        molecule._connections = self._connections.copy()
        return molecule

    # =========================================================================

    def center_of_mass(self) -> np.ndarray:
        """Computes the center of mass of the molecule."""
        weights = np.array([atom.weight for atom in self.atoms])
        return np.average(self._positions, weights=weights, axis=0)

    def central_atom(self) -> int:
        """Returns the index of the nearest atom to the center of mass."""
        mc = self.center_of_mass()
        dists = distances(self._positions, mc)
        return int(np.argmin(dists))

    def centered_molecule(self) -> 'Molecule':
        """Returns a copy of the molecule with its center of mass at the origin."""
        mc = self.center_of_mass()
        mol = self.copy()
        mol._positions -= mc
        return mol

    def get_atom(self, atom: Union[int, str, Atom]) -> Union[Atom, List[Atom]]:
        """Find an atom in the molecule.

        Parameters
        ----------
        atom: int or str or Atom
            The argument for getting the atom. If a ``int`` is passed
            it is interpreted as the index, if a ``str`` is passed as
            the name of an atom.

        Returns
        -------
        atom: Atom or list
        """
        if isinstance(atom, Atom):
            return atom
        elif isinstance(atom, int):
            return self._atoms[atom]
        else:
            atoms = list()
            for at in self._atoms:
                if atom == at.name:
                    atoms.append(at)
            if not atoms:
                raise ValueError(f"No Atom with the name '{atom}' found!")
            return atoms

    def get_alpha(self, atom: Union[int, str, Atom]) -> Union[int, List[int]]:
        """Returns the index of the atom in the unit-cell.

        Parameters
        ----------
        atom: int or str or Atom
            The argument for getting the atom. If a ``int`` is passed
            it is interpreted as the index, if a ``str`` is passed as
            the name of an atom.

        Returns
        -------
        alpha: int or list
        """
        if isinstance(atom, Atom):
            return self._atoms.index(atom)
        elif isinstance(atom, str):
            alphas = list()
            for i, at in enumerate(self._atoms):
                if atom == at.name:
                    alphas.append(i)
            return alphas
        return atom

    def get_atom_attrib(self, atom: Union[int, str, Atom], attrib: str,
                        default: Optional[Any] = None) -> Any:
        """ Returns an attribute of a specific atom in the unit cell.

        Parameters
        ----------
        atom: int or str or Atom
            Argument for finding an atom contained in the unit cell.
        attrib: str
            Name of the atom attribute.
        default: str or int or float or object, optional
            Default value used if the attribute couln't be found in the Atom dictionary.

        Returns
        -------
        attrib: object
        """
        atom = self.get_atom(atom)
        return atom.get(attrib, default)

    def get_atom_positions(self, atom: Union[int, str, Atom],
                           atleast2d: Optional[bool] = True) -> Iterator[np.ndarray]:
        """Return a list of all positions of a specific atom in the molecule.

        Parameters
        ----------
        atom: int or str or Atom
            Argument for finding an atom contained in the unit cell.
        atleast2d: bool, optional
            If ``True``, one-dimensional coordinates will be casted to 2D vectors.

        Yields
        -------
        positions: np.ndarray
        """
        atom = self.get_atom(atom)
        for at, pos in zip(self._atoms, self._positions):
            if atleast2d and len(pos) == 1:
                pos = np.array([pos[0], 0])
            if at == atom:
                yield pos

    def get_positions(self, atleast2d: Optional[bool] = True
                      ) -> Dict[Any, List[np.ndarray]]:
        """Returns a dict containing the positions of all unique atoms in the molecule.

        Parameters
        ----------
        atleast2d: bool, optional
            If ``True``, one-dimensional coordinates will be casted to 2D vectors.

        Returns
        -------
        atom_pos: dict of list
        """
        atom_pos = dict()
        for atom, pos in zip(self._atoms, self._positions):
            if atleast2d and len(pos) == 1:
                pos = np.array([pos[0], 0])

            if atom.name in atom_pos.keys():
                atom_pos[atom].append(pos)
            else:
                atom_pos[atom] = [pos]
        return atom_pos

    def get_neighbors(self, alpha: int = 0) -> List[int]:
        return list(self._connections[alpha])

    def get_neighbor_positions(self, alpha: int = 0) -> np.ndarray:
        return np.array([self._positions[i] for i in self._connections[alpha]])

    def get_neighbor_vectors(self, alpha: int = 0,
                             include_zero: Optional[bool] = False) -> np.ndarray:
        pos0 = self._positions[alpha]
        pos1 = self.get_neighbor_positions(alpha)
        if include_zero:
            pos1 = np.append(np.zeros((1, len(pos0))), pos1, axis=0)
        vecs = pos1 - pos0
        # logger.debug("Neighbour-vectors: %s", vecs)

        return vecs

    def get_neighbor_distance(self, alpha1: int, alpha2: int) -> float:
        return distance(self._positions[alpha1], self._positions[alpha2])

    def get_neighbor_distances(self, alpha: int = 0) -> np.ndarray:
        neighbors = self.get_neighbors(alpha)
        return np.array(
            [self.get_neighbor_distance(alpha, alpha2) for alpha2 in neighbors]
        )

    def translate(self, r: Union[float, Sequence[float]]) -> Iterator[np.ndarray]:
        """Translate the positions of the atoms contained in the molecule.

        Parameters
        ----------
        r: np.ndarray
            The vector for translating the positions.

        Yields
        -------
        positions: np.ndarray
        """
        r = np.atleast_1d(r)
        for atom, pos in zip(self._atoms, self._positions):
            yield r + pos

    def __copy__(self) -> 'Molecule':
        """Creates a deep copy of the ``UnitCell`` instance."""
        return self.copy()

    def __len__(self) -> int:
        """Return the number of atoms in the molecule."""
        return len(self._atoms)

    def __contains__(self, item: Union[Sequence, Atom]) -> bool:
        """Check if an atom or position is in the molecule."""
        if isinstance(item, Atom):
            return item in self._atoms
        else:
            item = np.asarray(item)
            for pos in self._positions:
                if np.all(item == pos):
                    return True
        return False

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the atoms contained in the molecule."""
        return iter(self._atoms)

    def __getitem__(self, item: int) -> Tuple[Atom, np.ndarray]:
        """Returns the ``Atom`` instance and position of a site"""
        return self._atoms[item], self.positions[item]

    def __dict__(self) -> Dict[Any, List[Union[np.ndarray, Any]]]:
        """Returns a dict of the positions of all unique atoms in the molecule."""
        return self.get_positions(atleast2d=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(atoms: {self.num_atoms})"

    def __str__(self) -> str:
        lines = [str(a) + ": " + str(p) for a, p in zip(self._atoms, self._positions)]
        return "\n".join(lines)
