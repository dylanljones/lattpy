Finite lattice models
---------------------

So far only abstract, infinite lattices have been discussed. In order to construct
a finite sized model of the configured lattice structure we have to build the lattice:

.. plot::
   :format: doctest
   :include-source:
   :context: close-figs

   >>> latt = lp.simple_square()
   >>> latt.build((10, 5))
   >>> latt.plot()
   >>> plt.show()


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
