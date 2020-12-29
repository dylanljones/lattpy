# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from .utils import (
    LatticeError,
    ConfigurationError,
    vrange,
    vlinspace,
    split_index,
    frmt_num,
    frmt_time,
    Timer
)

from .spatial import (
    distance,
    cell_size,
    cell_volume,
    compute_vectors,
    compute_neighbours,
    VoronoiTree,
    WignerSeitzCell
)

from .unitcell import (
    Atom,
    UnitCell
)

from .data import LatticeData
from .disptools import DispersionPath
from .lattice import Lattice

# =========================================================================
#                             1D Lattices
# =========================================================================


def simple_chain(a=1.0, atom=None, neighbours=1):
    latt = Lattice.chain(a)
    latt.add_atom(atom=atom)
    latt.calculate_distances(neighbours)
    return latt


def alternating_chain(a=1.0, atom1=None, atom2=None, x0=0.0, neighbours=1):
    latt = Lattice.chain(a)
    latt.add_atom(pos=(0.0 + x0) * a, atom=atom1)
    latt.add_atom(pos=(0.5 + x0) * a, atom=atom2)
    latt.calculate_distances(neighbours)
    return latt


# =========================================================================
#                             2D Lattices
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
#                             3D Lattices
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
