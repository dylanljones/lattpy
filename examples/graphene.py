# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from lattpy import Atom, Lattice
import matplotlib.pyplot as plt

A = 1.0
ATOM_1 = Atom("C_1", color="C0")
ATOM_2 = Atom("C_2", color="C0")

# Construct graphene lattice with nearest neighbors
latt = Lattice.hexagonal(A)
latt.add_atom([0, 0], atom=ATOM_1)
latt.add_atom([A, 0], atom=ATOM_2)
latt.set_num_neighbors(1)

# Build the lattice in a square
latt.build((10, 10))

# Plot the lattice
ax = latt.plot(show=False, legend=False)
ax.set_title("Graphene Lattice")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.get_figure().tight_layout()

# Plot 1. Brillouin zone of lattice
bz = latt.brillouin_zone()

ax = bz.draw(color="r")
ax.set_title("1. Brillouin zone of graphene")
ax.set_xlabel("$k_x$")
ax.set_ylabel("$k_y$")

plt.show()
