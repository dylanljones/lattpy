# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from .utils import (
    logger,
    ArrayLike,
    LatticeError,
    ConfigurationError,
    min_dtype,
    chain,
    frmt_num,
    frmt_bytes,
    frmt_time,
    Timer
)

from .spatial import (
    distance,
    interweave,
    vindices,
    vrange,
    cell_size,
    cell_volume,
    compute_vectors,
    compute_neighbors,
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

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# =========================================================================
#                             1D Lattices
# =========================================================================


def simple_chain(a=1.0, atom=None, neighbors=1):
    latt = Lattice.chain(a)
    latt.add_atom(atom=atom)
    latt.set_num_neighbors(neighbors)
    return latt


def alternating_chain(a=1.0, atom1=None, atom2=None, x0=0.0, neighbors=1):
    latt = Lattice.chain(a)
    latt.add_atom(pos=(0.0 + x0) * a, atom=atom1)
    latt.add_atom(pos=(0.5 + x0) * a, atom=atom2)
    latt.set_num_neighbors(neighbors)
    return latt


# =========================================================================
#                             2D Lattices
# =========================================================================

def simple_square(a=1.0, atom=None, neighbors=1):
    latt = Lattice.square(a)
    latt.add_atom(atom=atom)
    latt.set_num_neighbors(neighbors)
    return latt


def simple_rectangular(a1=1.0, a2=1.0, atom=None, neighbors=1):
    latt = Lattice.rectangular(a1, a2)
    latt.add_atom(atom=atom)
    latt.set_num_neighbors(neighbors)
    return latt


def centered_rectangular(a1=1.0, a2=1.0, atom=None, neighbors=1):
    latt = Lattice([[a1, 0], [a1/2, a2/2]])
    latt.add_atom(atom=atom)
    latt.set_num_neighbors(neighbors)
    return latt


def graphene(a=1.0, neighbors=1):
    at = Atom("C")
    latt = Lattice.hexagonal(a)
    latt.add_atom([0, 0], at)
    latt.add_atom([a, 0], at)
    latt.set_num_neighbors(neighbors)
    return latt


# =========================================================================
#                             3D Lattices
# =========================================================================

def simple_cubic(a=1.0, atom=None, neighbors=1):
    latt = Lattice.sc(a)
    latt.add_atom(atom=atom)
    latt.set_num_neighbors(neighbors)
    return latt


def nacl_structure(a=1.0, atom1="Na", atom2="Cl", neighbors=1):
    latt = Lattice.fcc(a)
    latt.add_atom(pos=[0, 0, 0], atom=atom1)
    latt.add_atom(pos=[a/2, a/2, a/2], atom=atom2)
    latt.set_num_neighbors(neighbors)
    return latt
