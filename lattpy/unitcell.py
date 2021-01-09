# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Objects for representing atoms and the unitcell of a lattice."""

import itertools
import collections
import numpy as np
from typing import Union, Optional, Any, Iterator, Dict, Sequence, List, Tuple
from .utils import SiteOccupiedError

__all__ = ["Atom", "UnitCell"]


class Atom(collections.abc.MutableMapping):
    """Object representing an atom of a bravais lattice."""

    _counter = itertools.count()

    __slots__ = ["_index", "_name", "_params"]

    def __init__(self, name: Optional[str] = None, color: Optional[str] = None,
                 size: Optional[int] = 10, **kwargs):
        super().__init__()
        index = next(Atom._counter)
        self._index = index
        self._name = name or str(index)
        self._params = dict(color=color, size=size, **kwargs)

    @property
    def index(self) -> int:
        """Return the index of the ``Atom`` instance."""
        return self._index

    @property
    def name(self) -> str:
        """Return the name of the ``Atom`` instance."""
        return self._name

    def dict(self) -> Dict[str, Any]:
        """Returns the data of the ``Atom`` instance as a dictionary."""
        data = dict(index=self._index, name=self._name)
        data.update(self._params)
        return data

    def copy(self) -> 'Atom':
        """Creates a deep copy of the ``Atom`` instance."""
        return Atom(self.name, **self._params.copy())

    def get(self, key: str, default=None) -> Any:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __len__(self) -> int:
        """Return the length of the ``Atom`` attributes."""
        return len(self._params)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys of the ``Atom`` attributes."""
        return iter(self._params)

    def __getitem__(self, key: str) -> Any:
        """Make ``Atom`` attributes accessable as dictionary items."""
        return self._params[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Make ``Atom`` attributes accessable as dictionary items."""
        self._params[key] = value

    def __delitem__(self, key: str) -> None:
        """Make ``Atom`` attributes accessable as dictionary items."""
        del self._params[key]

    def __getattribute__(self, key: str) -> Any:
        """Make ``Atom`` attributes accessable as attributes."""
        key = str(key)
        if not key.startswith("_") and key in self._params.keys():
            return self._params[key]
        else:
            return super().__getattribute__(key)

    def __setattr__(self, key: str, value: Any) -> None:
        """Make ``Atom`` attributes accessable as attributes."""
        key = str(key)
        if not key.startswith("_") and key in self._params.keys():
            self._params[key] = value
        else:
            super().__setattr__(key, value)

    def __hash__(self) -> hash:
        """Make ``Atom`` instance hashable."""
        return hash(self._name)

    def __dict__(self) -> Dict[str, Any]:
        """Return the information of the atom as a dictionary"""
        return self.dict()

    def __copy__(self) -> 'Atom':
        """Creates a deep copy of the ``Atom`` instance."""
        return self.copy()

    def __eq__(self, other: Union['Atom', str]) -> bool:
        if isinstance(other, Atom):
            return self._name == other._name
        else:
            return self._name == other

    def __repr__(self) -> str:
        argstr = f"{self._name}"
        paramstr = ", ".join(f"{k}={v}" for k, v in self._params.items() if v)
        if paramstr:
            argstr += ", " + paramstr
        return f"Atom({argstr})"


class UnitCell(collections.abc.Sequence):
    """``Atom`` container representing the unitcell of a bravais lattice."""

    __slots__ = ["_num_base", "_atoms", "_positions"]

    def __init__(self):
        """Initialize a unitcell instance."""
        super().__init__()
        self._num_base = 0
        self._atoms = list()
        self._positions = list()

    @property
    def num_base(self):
        """Return the number of atoms in the unitcell."""
        return self._num_base

    @property
    def atoms(self):
        """Return the atoms contained in the unitcell."""
        return self._atoms

    @property
    def positions(self):
        """Return the positions of the atoms contained in the unitcell."""
        return np.asarray(self._positions)

    def add(self, pos: Optional[Union[float, Sequence[float]]] = None,
            atom: Optional[Union[str, Dict[str, Any], Atom]] = None,
            **kwargs) -> Atom:
        """ Adds a new atom to the unitcell.

        Raises
        ------
        ValueError:
            A Value Error is raised if the position of the new atom is already occupied.

        Parameters
        ----------
        pos: (N) array_like or float, optional
            Position of site in the unit-cell. The default is the origin of the cell.
            The size of the array has to match the dimension of the lattice.
        atom: str or dict or Atom, optional
            Identifier of the site. If a string is passed, a new Atom instance is created.
        **kwargs
            Keyword arguments for ´Atom´ constructor. Only used if a new Atom instance is created.

        Returns
        -------
        atom: Atom
        """
        pos = np.atleast_1d(pos)
        if any(np.all(pos == x) for x in self._positions):
            raise SiteOccupiedError(atom, pos)

        if not isinstance(atom, Atom):
            atom = Atom(atom, **kwargs)

        self._atoms.append(atom)
        self._positions.append(np.asarray(pos))

        # Update number of base atoms if data is valid
        assert len(self._atoms) == len(self._positions)
        self._num_base = len(self._positions)
        return atom

    def remove(self, index: int) -> None:
        """Removes an existing atom from the unitcell.

        Parameters
        ----------
        index: int
            The indx of the atom to remove.
        """
        if index >= len(self._atoms):
            raise ValueError(f"Index {index} out of range for unitcell "
                             f"with {len(self._atoms)} atoms!")
        del self._atoms[index]
        del self._positions[index]

    def zip(self) -> Iterator[Tuple[Atom, np.ndarray]]:
        """Iterate over the atoms and positions in the unitcell."""
        return zip(self._atoms, self._positions)

    def get_atom(self, atom: Union[int, str, Atom]) -> Atom:
        """Find an atom in the unitcell.

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
        """Return a list of all positions of a specific atom in the unitcell.

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

    def get_positions(self, atleast2d: Optional[bool] = True) -> Dict[Any, List[np.ndarray]]:
        """Returns a dict containing the positions of all unique atoms in the unitcell.

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

    def translate(self, r: Union[float, Sequence[float]]) -> Iterator[np.ndarray]:
        """Translate the positions of the atoms contained in the unitcell.

        Parameters
        ----------
        r: np.ndarray
            The vector for translating the positions.

        Yields
        -------
        positions: np.ndarray
        """
        r = np.atleast_1d(r)
        for atom, pos in self.zip():
            yield r + pos

    def copy(self) -> 'UnitCell':
        """Creates a deep copy of the ``UnitCell`` instance."""
        cell = UnitCell()
        for pos, atom in zip(self._positions, self._atoms):
            cell.add(pos.copy(), atom.copy())
        return cell

    def __copy__(self) -> 'UnitCell':
        """Creates a deep copy of the ``UnitCell`` instance."""
        return self.copy()

    def __len__(self) -> int:
        """Return the number of atoms in the unitcell."""
        return len(self._atoms)

    def __contains__(self, item: Union[Sequence, Atom]) -> bool:
        """Check if an atom or position is in the unitcell."""
        if isinstance(item, Atom):
            return item in self._atoms
        else:
            item = np.asarray(item)
            for pos in self._positions:
                if np.all(item == pos):
                    return True
        return False

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the atoms contained in the unitcell."""
        return iter(self._atoms)

    def __getitem__(self, item: int) -> Tuple[Atom, np.ndarray]:
        """Returns the ``Atom`` instance and position of a site"""
        return self._atoms[item], self.positions[item]

    def __dict__(self) -> Dict[Any, List[Union[np.ndarray, Any]]]:
        """Returns a dict containing the positions of all unique atoms in the unitcell"""
        return self.get_positions(atleast2d=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(atoms: {self.num_base})"

    def __str__(self) -> str:
        lines = [str(a) + ": " + str(p) for a, p in zip(self._atoms, self._positions)]
        return "\n".join(lines)
