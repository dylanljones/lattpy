# coding: utf-8
"""
Created on 05 Apr 2020
author: Dylan Jones
"""
from .core import *
from .bravais import BravaisLattice
from .lattice import Lattice


def simple_chain(a=1.0, atom=None, neighbours=1):
    latt = Lattice.chain(a)
    latt.add_atom(atom=atom)
    latt.calculate_distances(neighbours)
    return latt


# =========================================================================
#                             2D Prefabs
# =========================================================================

def simple_square(a=1.0, atom=None, neighbours=1):
    latt = Lattice.square(a)
    latt.add_atom(atom=atom)
    latt.calculate_distances(neighbours)
    return latt


def simple_rectangular(a1=1.0, a2=1.0, atom=None, neighbours=1):
    latt = Lattice.rectangular(a1, a2)
    latt.add_atom(atom=atom)
    latt.calculate_distances(neighbours)
    return latt


def centered_rectangular(a1=1.0, a2=1.0, atom=None, neighbours=1):
    latt = Lattice([[a1, 0], [a1/2, a2/2]])
    latt.add_atom(atom=atom)
    latt.calculate_distances(neighbours)
    return latt


def graphene(a=1.0, neighbours=1):
    at = Atom("C")
    latt = Lattice.hexagonal(a)
    latt.add_atom([0, 0], at)
    latt.add_atom([a, 0], at)
    latt.calculate_distances(neighbours)
    return latt


# =========================================================================
#                             3D Prefabs
# =========================================================================

def simple_cubic(a=1.0, atom=None, neighbours=1):
    latt = Lattice.sc(a)
    latt.add_atom(atom=atom)
    latt.calculate_distances(neighbours)
    return latt


def nacl_structure(a=1.0, atom1="Na", atom2="Cl", neighbours=1):
    latt = Lattice.fcc(a)
    latt.add_atom(pos=[0, 0, 0], atom=atom1)
    latt.add_atom(pos=[a/2, a/2, a/2], atom=atom2)
    latt.calculate_distances(neighbours)
    return latt
