# lattpy 0.1

`lattpy` is a python package for modeling bravais lattices and constructing (finite) lattice structures.

Installation
------------

Download package and install via pip
````commandline
pip install -e <folder path>
````
or the setup.py script
````commandline
python setup.py --install
````


Usage
=====

Before accessing the attributes of the `Lattice`-model the lattice has to be configured

Configuration
-------------

A new instance of a lattice model is initialized using the unit-vectors of the Bravais lattice.
After the initialization the atoms of the unit-cell need to be added. To finish the configuration
the number of distances in the lattice need to be set. This computes the nearest distances between
all atoms of the unit-cells. If only the nearest distance is computed the lattice will be set to 
nearest neighbours.
````python
import numpy as np
from lattpy import Lattice

latt = Lattice(np.eye(2))       # Construct a Bravais lattice with square unit-vectors
latt.add_atom(pos=[0.0, 0.0])   # Add an Atom to the unit cell of the lattice
latt.calculate_distances(1)     # Set the maximum number of distances in the configuration.
````

To speed up the configuration prefabs of common lattices are included. The previous lattice for example
can also be constructed as following:
````python
from lattpy import simple_square

latt = simple_square(a=1.0, neighbours=1)  # Initializes a square lattice with one atom in the unit-cell
````

So far only the lattice structure has been configured. To actually construct a (finite) model of the lattice
the model has to be buildt:
````python
from lattpy import simple_square

latt = simple_square(a=1.0)  # Initializes a square lattice with one atom in the unit-cell
latt.build(shape=(5, 2))
````
This will compute the indices and neighbours of all sites in the given shape and store the data.
By default the lattice is buildt in real-space, meaning the shape parameter passed to the `build`-method is
interpreted as a rectangle in real space. Alternatively the lattice can be buildt in the unit-vector-space:
````python
from lattpy import simple_square

latt = simple_square(a=1.0)  # Initializes a square lattice with one atom in the unit-cell
latt.build(shape=(5, 2), inbound=False)
````

After building the lattice periodic boundary conditions can be set along one or multiple axes:
````python
from lattpy import simple_square

latt = simple_square(a=1.0)  # Initializes a square lattice with one atom in the unit-cell
latt.build(shape=(5, 2), inbound=False)

latt.set_periodic(axis=0)
# or
latt.set_periodic(axis=[0, 1])
````



Examples
========

Using the (buildt) lattice model it is easy to construct the (kinetic) Hamiltonian of the model:

````python
import numpy as np
from lattpy import simple_chain

# Initializes a 1D lattice chain with a length of 5 atoms.
latt = simple_chain(a=1.0)
latt.build(shape=5, inbound=False)
n = latt.n_sites

# Construct the non-interacting (kinetic) Hamiltonian-matrix
eps, t = 0, 1
ham = np.zeros((n, n))
for i in range(n):
    ham[i, i] = eps
    for j in latt.nearest_neighbours(i, unique=True):
        ham[i, j] = t
        ham[j, i] = t
````
This will create the Hamiltonian-matrix:
````
[[0. 1. 0. 0. 0.]
 [1. 0. 1. 0. 0.]
 [0. 1. 0. 1. 0.]
 [0. 0. 1. 0. 1.]
 [0. 0. 0. 1. 0.]]
````

If periodic boundary conditions are set (along axis 0) the matrix will be:
````
[[0. 1. 0. 0. 1.]
 [1. 0. 1. 0. 0.]
 [0. 1. 0. 1. 0.]
 [0. 0. 1. 0. 1.]
 [1. 0. 0. 1. 0.]]
````
