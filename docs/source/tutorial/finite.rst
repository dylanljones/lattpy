Finite lattice models
---------------------

So far only abstract, infinite lattices have been discussed. In order to construct
a finite sized model of the configured lattice structure we have to build the lattice.

Build geometries
~~~~~~~~~~~~~~~~

By default, the shape passed to the ``build`` is used to create a box in cartesian
coordinates. Alternatively, the geometry can be constructed in the basis of the lattice
by setting ``primitive=True``. As an example, consider the hexagonal lattice. We can
build the lattice in a box of the specified shape:

.. plot::
   :format: doctest
   :include-source:
   :context: close-figs

   >>> latt = lp.Lattice.hexagonal()
   >>> latt.add_atom()
   >>> latt.add_connections()
   >>> s = latt.build((10, 10))
   >>> ax = latt.plot()
   >>> s.plot(ax)
   >>> plt.show()


or in the coordinate system of the lattice, which results in


.. plot::
   :format: doctest
   :include-source:
   :context: close-figs

   >>> latt = lp.Lattice.hexagonal()
   >>> latt.add_atom()
   >>> latt.add_connections()
   >>> s = latt.build((10, 10), primitive=True)
   >>> ax = latt.plot()
   >>> s.plot(ax)
   >>> plt.show()



Other geometries can be build by using ``AbstractShape`` ojects:

.. plot::
   :format: doctest
   :include-source:
   :context: close-figs

   >>> latt = lp.Lattice.hexagonal()
   >>> latt.add_atom()
   >>> latt.add_connections()
   >>> s = lp.Circle((0, 0), radius=10)
   >>> latt.build(s, primitive=True)
   >>> ax = latt.plot()
   >>> s.plot(ax)
   >>> plt.show()


Periodic boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a finite size lattice model has been buildt periodic boundary conditions can
be configured by specifying the axis of the periodic boundary conditions.
The periodic boundary conditions can be set up for each axes individually, for example

.. plot::
   :format: doctest
   :include-source:
   :context: close-figs

   >>> latt = lp.simple_square()
   >>> latt.build((6, 4))
   >>> latt.set_periodic(0)
   >>> latt.plot()
   >>> plt.show()

or for multiple axes at once:

.. plot::
   :format: doctest
   :include-source:
   :context: close-figs

   >>> latt = lp.simple_square()
   >>> latt.build((6, 4))
   >>> latt.set_periodic([0, 1])
   >>> latt.plot()
   >>> plt.show()


As before the axis can be assumed to be in world or lattice coordinates by the
``primitive`` keyword. If ``primitive=False``, i.e. world coordinates, the box around
the buildt lattice is reoeated periodically:

.. plot::
   :format: doctest
   :include-source:
   :context: close-figs

   >>> latt = lp.graphene()
   >>> latt.build((5.5, 4.5))
   >>> latt.set_periodic(0)
   >>> latt.plot()
   >>> plt.show()

Here, the periodic boundary conditions again are set up along the x-axis, even though
the basis vectors of the hexagonal lattice define a new basis. If the periodic
boundary conditions should be set up along one of the basis vectors ``primitive=True``
is used:

.. plot::
   :format: doctest
   :include-source:
   :context: close-figs

   >>> latt = lp.graphene()
   >>> latt.build((5.5, 4.5), primitive=True)
   >>> latt.set_periodic(0, primitive=True)
   >>> latt.plot()
   >>> plt.show()

.. warning::
   The ``set_periodic`` method assumes the lattice is build such that periodic
   boundary condtitions are possible. This is especially important if a lattice
   with multiple atoms in the unit cell is used. To correctly connect both sides of
   the lattice it has to be ensured that each cell in the lattice is fully contained.
   If, for example, the last unit cell in the x-direction is cut off in the middle
   no perdiodic boundary conditions will be computed since the distance between the
   two edges is larger than the other distances in the lattice.
   A future version will check if this requirement is fulfilled, but until now the
   user is responsible for the correct configuration.



Position and neighbor data
~~~~~~~~~~~~~~~~~~~~~~~~~~

After building the lattice and optionally setting periodic boundary conditions the
information of the buildt lattice can be accessed. The data of the
lattice model then can be accessed by a simple index ``i``. The syntax is the same as
before, just without the ``get_`` prefix. In order to find the right index,
the ``plot`` method also supports showing the coorespnding super indices of the lattice sites:

.. plot::
   :format: doctest
   :include-source:
   :context: close-figs

   >>> latt = lp.simple_square()
   >>> latt.build((6, 4))
   >>> latt.set_periodic(0)
   >>> latt.plot(show_indices=True)
   >>> plt.show()

The positions of the sites in the model can now be accessed via the super index ``i``:

>>> latt.position(2)
[0. 2.]

Similarly, the neighbors can be found via

>>> latt.neighbors(2, distidx=0)
[3 1 7 32]

The nearest neighbors also can be found with the helper method

>>> latt.nearest_neighbors(2)
[3 1 7 32]


The position and neighbor data of the finite lattice model is stored in the
``LatticeData`` object, wich can be accessed via the ``data`` attribute.
Additionally, the positions and (lattice) indices of the model can be directly
fetched, for example

>>> latt.positions
[[0. 0.]
 [0. 1.]
 ...
 [6. 3.]
 [6. 4.]]
