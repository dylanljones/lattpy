Finite lattice data
-------------------

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
