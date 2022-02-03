.. lattpy documentation master file, created by
   sphinx-quickstart on Mon Jan 31 12:11:09 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================================
Welcome to LattPy's documentation!
==================================

|pypi-python-version| |pypi-version| |pypi-status| |pypi-license|

+---------+---------------+-----------------+
| Master  ||tests-master| ||codecov-master| |
+---------+---------------+-----------------+
| Dev     ||tests-dev|    ||codecov-dev|    |
+---------+---------------+-----------------+


LattPy is a simple and efficient Python package for modeling Bravais lattices and
constructing (finite) lattice structures in ``d`` dimensions.

.. plot::
   :format: doctest
   :context: close-figs

   >>> latt = lp.graphene()
   >>> latt.build((20.7, 10.5))
   >>> latt.plot(legend=False)
   >>> plt.show()


.. warning::
   This project is still in development and might change significantly in the future!


.. toctree::
   :maxdepth: 3
   :caption: User Guide

   installation
   quickstart
   tutorial/index


.. toctree::
   :maxdepth: 1
   :caption: API Reference

   generated/lattpy
   generated/lattpy.data
   generated/lattpy.disptools
   generated/lattpy.lattice
   generated/lattpy.plotting
   generated/lattpy.shape
   generated/lattpy.spatial
   generated/lattpy.unitcell
   generated/lattpy.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |pypi-python-version| image:: https://img.shields.io/pypi/pyversions/lattpy?logo=python&style=flat-square
   :alt: PyPI - Python Version
.. |pypi-version| image:: https://img.shields.io/pypi/v/lattpy?logo=pypi&style=flat-square
   :alt: PyPI - Version
.. |pypi-status| image:: https://img.shields.io/pypi/status/lattpy?color=yellow&style=flat-square
   :alt: PyPI - Status
.. |pypi-license| image:: https://img.shields.io/pypi/l/lattpy?style=flat-square
   :alt: PyPI - License
.. |tests-master| image:: https://img.shields.io/github/workflow/status/dylanljones/lattpy/Test/master?label=tests&logo=github&style=flat
   :alt: Test status master
   :target: https://github.com/dylanljones/lattpy/actions/workflows/test.yml
.. |tests-dev| image:: https://img.shields.io/github/workflow/status/dylanljones/lattpy/Test/dev?label=tests&logo=github&style=flat
   :alt: Test status dev
   :target: https://github.com/dylanljones/lattpy/actions/workflows/test.yml
.. |codecov-master| image:: https://codecov.io/gh/dylanljones/lattpy/branch/master/graph/badge.svg?token=P61R3IQKXC
   :alt: Coverage master
   :target: https://app.codecov.io/gh/dylanljones/lattpy/branch/master
.. |codecov-dev| image:: https://codecov.io/gh/dylanljones/lattpy/branch/dev/graph/badge.svg?token=P61R3IQKXC
   :alt: Coverage dev
   :target: https://app.codecov.io/gh/dylanljones/lattpy/branch/dev
