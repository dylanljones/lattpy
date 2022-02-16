# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Package for modeling Bravais lattices and finite lattice structures.

Submodules
----------

.. autosummary::
   lattpy.atom
   lattpy.data
   lattpy.disptools
   lattpy.lattice
   lattpy.shape
   lattpy.spatial
   lattpy.utils

"""

from .utils import (
    logger,
    ArrayLike,
    LatticeError,
    ConfigurationError,
    SiteOccupiedError,
    NoConnectionsError,
    NotAnalyzedError,
    NotBuiltError,
    min_dtype,
    chain,
    frmt_num,
    frmt_bytes,
    frmt_time,
)

from .spatial import (
    distance,
    distances,
    interweave,
    vindices,
    vrange,
    cell_size,
    cell_volume,
    compute_vectors,
    VoronoiTree,
    WignerSeitzCell
)

from .shape import AbstractShape, Shape, Circle, Donut, ConvexHull
from .data import LatticeData
from .disptools import DispersionPath
from .atom import Atom
from .lattice import Lattice

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# =========================================================================
#                             1D Lattices
# =========================================================================


def simple_chain(a=1.0, atom=None, neighbors=1):
    """Creates a 1D lattice with one atom at the origin of the unit cell.

    Parameters
    ----------
    a : float, optional
        The lattice constant (length of the basis-vector).
    atom : str or Atom, optional
        The atom to add to the lattice. If a string is passed, a new ``Atom`` instance
        is created.
    neighbors : int, optional
        The number of neighbor-distance levels, e.g. setting to 1 means only
        nearest neighbors. The default is nearest neighbors (1).

    Returns
    -------
    latt : Lattice
        The configured lattice instance.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> latt = lp.simple_chain()
    >>> latt.plot_cell()
    >>> plt.show()

    """
    latt = Lattice.chain(a)
    latt.add_atom(atom=atom)
    latt.add_connections(neighbors)
    return latt


def alternating_chain(a=1.0, atom1=None, atom2=None, x0=0.0, neighbors=1):
    """Creates a 1D lattice with two atoms in the unit cell.

    Parameters
    ----------
    a : float, optional
        The lattice constant (length of the basis-vector).
    atom1 : str or Atom, optional
        The first atom to add to the lattice. If a string is passed, a new
        ``Atom`` instance is created.
    atom2 : str or Atom, optional
        The second atom to add to the lattice. If a string is passed, a new
        ``Atom`` instance is created.
    x0 : float, optional
        The offset of the atom positions in x-direction.
    neighbors : int, optional
        The number of neighbor-distance levels, e.g. setting to 1 means only
        nearest neighbors. The default is nearest neighbors (1).

    Returns
    -------
    latt : Lattice
        The configured lattice instance.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> latt = lp.alternating_chain()
    >>> latt.plot_cell()
    >>> plt.show()

    """
    latt = Lattice.chain(a)
    latt.add_atom(pos=(0.0 + x0) * a, atom=atom1)
    latt.add_atom(pos=(0.5 + x0) * a, atom=atom2)
    latt.add_connections(neighbors)
    return latt


# =========================================================================
#                             2D Lattices
# =========================================================================

def simple_square(a=1.0, atom=None, neighbors=1):
    """Creates a square lattice with one atom at the origin of the unit cell.

    Parameters
    ----------
    a : float, optional
        The lattice constant (length of the basis-vector).
    atom : str or Atom, optional
        The atom to add to the lattice. If a string is passed, a new ``Atom`` instance
        is created.
    neighbors : int, optional
        The number of neighbor-distance levels, e.g. setting to 1 means only
        nearest neighbors. The default is nearest neighbors (1).

    Returns
    -------
    latt : Lattice
        The configured lattice instance.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> latt = lp.simple_square()
    >>> latt.plot_cell()
    >>> plt.show()

    """
    latt = Lattice.square(a)
    latt.add_atom(atom=atom)
    latt.add_connections(neighbors)
    return latt


def simple_rectangular(a1=1.5, a2=1.0, atom=None, neighbors=2):
    """Creates a rectangular lattice with one atom at the origin of the unit cell.

    Parameters
    ----------
    a1 : float, optional
        The lattice constant in the x-direction.
    a2 : float, optional
        The lattice constant in the y-direction.
    atom : str or Atom, optional
        The atom to add to the lattice. If a string is passed, a new ``Atom`` instance
        is created.
    neighbors : int, optional
        The number of neighbor-distance levels, e.g. setting to 1 means only
        nearest neighbors. The default is nearest neighbors (1).

    Returns
    -------
    latt : Lattice
        The configured lattice instance.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> latt = lp.simple_rectangular()
    >>> latt.plot_cell()
    >>> plt.show()

    """
    latt = Lattice.rectangular(a1, a2)
    latt.add_atom(atom=atom)
    latt.add_connections(neighbors)
    return latt


def graphene(a=1.0):
    """Creates a hexagonal lattice with two atoms in the unit cell.

    Parameters
    ----------
    a : float, optional
        The lattice constant (length of the basis-vectors).

    Returns
    -------
    latt : Lattice
        The configured lattice instance.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> latt = lp.graphene()
    >>> latt.plot_cell()
    >>> plt.show()

    """
    at1 = Atom("C1")
    at2 = Atom("C2")
    latt = Lattice.hexagonal(a)
    latt.add_atom([0, 0], at1)
    latt.add_atom([a, 0], at2)
    latt.add_connection(at1, at2, analyze=True)
    return latt


# =========================================================================
#                             3D Lattices
# =========================================================================

def simple_cubic(a=1.0, atom=None, neighbors=1):
    """Creates a cubic lattice with one atom at the origin of the unit cell.

    Parameters
    ----------
    a : float, optional
        The lattice constant (length of the basis-vector).
    atom : str or Atom, optional
        The atom to add to the lattice. If a string is passed, a new ``Atom`` instance
        is created.
    neighbors : int, optional
        The number of neighbor-distance levels, e.g. setting to 1 means only
        nearest neighbors. The default is nearest neighbors (1).

    Returns
    -------
    latt : Lattice
        The configured lattice instance.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> latt = lp.simple_cubic()
    >>> latt.plot_cell()
    >>> plt.show()

    """
    latt = Lattice.sc(a)
    latt.add_atom(atom=atom)
    latt.add_connections(neighbors)
    return latt


def nacl_structure(a=1.0, atom1="Na", atom2="Cl", neighbors=1):
    """Creates a NaCl lattice structure.

    Parameters
    ----------
    a : float, optional
        The lattice constant (length of the basis-vector).
    atom1 : str or Atom, optional
        The first atom to add to the lattice. If a string is passed, a new
        ``Atom`` instance is created. The default name is `Na`.
    atom2 : str or Atom, optional
        The second atom to add to the lattice. If a string is passed, a new
        ``Atom`` instance is created.. The default name is `Cl`.
    neighbors : int, optional
        The number of neighbor-distance levels, e.g. setting to 1 means only
        nearest neighbors. The default is nearest neighbors (1).

    Returns
    -------
    latt : Lattice
        The configured lattice instance.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> latt = lp.nacl_structure()
    >>> latt.plot_cell()
    >>> plt.show()

    """
    latt = Lattice.fcc(a)
    latt.add_atom(pos=[0, 0, 0], atom=atom1)
    latt.add_atom(pos=[a / 2, a / 2, a / 2], atom=atom2)
    latt.add_connections(neighbors)
    return latt


# ======================================================================================
# Other
# ======================================================================================


def finite_hypercubic(s, a=1.0, atom=None, neighbors=1):
    """Creates a d-dimensional finite lattice model with one atom in the unit cell.

    Parameters
    ----------
    s : float or Sequence[float] or AbstractShape
        The shape of the finite lattice. This also defines the dimensionality.
    a : float, optional
        The lattice constant (length of the basis-vectors).
    atom : str or Atom, optional
        The atom to add to the lattice. If a string is passed, a new ``Atom`` instance
        is created.
    neighbors : int, optional
        The number of neighbor-distance levels, e.g. setting to 1 means only
        nearest neighbors. The default is nearest neighbors (1).

    Returns
    -------
    latt : Lattice
        The configured lattice instance.

    Examples
    --------
    Simple chain:

    >>> import matplotlib.pyplot as plt
    >>> latt = lp.finite_hypercubic(4)
    >>> latt.plot()
    >>> plt.show()

    Simple square:

    >>> import matplotlib.pyplot as plt
    >>> latt = lp.finite_hypercubic((4, 2))
    >>> latt.plot()
    >>> plt.show()

    """
    import numpy as np
    if isinstance(s, (float, int)):
        s = (s, )

    dim = s.dim if isinstance(s, AbstractShape) else len(s)
    latt = Lattice(a * np.eye(dim))
    latt.add_atom(atom=atom)
    latt.add_connections(neighbors)
    latt.build(s)
    return latt
