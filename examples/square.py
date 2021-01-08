# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from lattpy import Lattice
import matplotlib.pyplot as plt

# Construct a square lattice structure
latt = Lattice.square(a=1.0)

# Add an Atom to the unitcell of the lattice and set nearest neighbours
latt.add_atom(pos=[0, 0], neighbours=1)

# Build the lattice
latt.build((5, 3))

# Plot the lattice
ax = latt.plot(show=False)
ax.set_title("Square Lattice")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

# Plot 1. Brillouin zone of lattice
bz = latt.brillouin_zone()

ax = bz.draw(color="r")
ax.set_title("1. Brillouin zone")
ax.set_xlabel("$k_x$")
ax.set_ylabel("$k_y$")

plt.show()
