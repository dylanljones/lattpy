# coding: utf-8
"""
Created on 12 May 2020
author: Dylan Jones
"""
import itertools
import numpy as np


class Atom:

    COUNTER = itertools.count()

    def __init__(self, name=None, color=None, size=10, **kwargs):
        idx = next(self.COUNTER)
        name = name or str(idx)
        self.name = name
        self.col = color
        self.size = size
        self.kwargs = kwargs

    def __getitem__(self, item):
        return self.kwargs[item]

    def attrib(self, key, default=None):
        return self.kwargs.get(key, default)

    def label(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Atom):
            return self.name == other.name
        else:
            return self.name == other

    def __repr__(self):
        return f"Atom({self.name}, {self.col}, {self.size})"


class UnitCell:

    def __init__(self):
        self.n_base = 0
        self.atoms = list()
        self.atom_positions = list()

    def add_atom(self, pos, atom=None, **kwargs):
        pos = np.asarray(pos)
        if any(np.all(pos == x) for x in self.atom_positions):
            raise ValueError(f"The position {pos} in the unit-cell already occupied!")
        if isinstance(atom, Atom):
            atom = atom
        else:
            atom = Atom(atom, **kwargs)
        self.atoms.append(atom)
        self.atom_positions.append(np.asarray(pos))
        self.n_base = len(self.atom_positions)
        return atom

    def unique(self):
        atom_pos = dict()
        for atom, pos in zip(self.atoms, self.atom_positions):
            if atom.name in atom_pos.keys():
                atom_pos[atom].append(pos)
            else:
                atom_pos[atom] = [pos]
        return atom_pos
