# What's New

<a name="0.7.3"></a>
## [0.7.3] - 2022-06-05
### Improvements/Bug Fixes
- `adjacency_matrix` is now vectorized and returns a `csr_matrix`
- passing a False boolean as axis to `set_periodic` now removes the periodic boundaries


<a name="0.7.2"></a>
## [0.7.2] - 2022-04-05
### New Features
- add prefabs for the hexagonal (triangular) and honeycomb lattice.
- add methods for building sparse matrices to `DataMap` class

### Improvements/Bug Fixes
- add argument for building in primitive basis to the `finite_hypercubic` method.


<a name="0.7.1"></a>
## [0.7.1] - 2022-29-03
### New Features
- add argument for setting periodic boundary conditions to the `finite_hypercubic` method.
- add method for computing minimum distances in a lattice with periodic boundary conditions
- add `shape` keyword argument to Lattice constructor
- add CSR/BSR sparse matrix format of indices and index-pointers to DataMap

### Code Refactoring
- rename `distance` variables to `distances_` to prevent same name as method


<a name="0.7.0"></a>
## [0.7.0] - 2022-21-02
### New Features
- Add method for computing the adjacency matrix of the lattice graph
- Split the lattice structure into separate object ``LatticeStructure`` and use it as base class for ``Lattice``
- Split the lattice basis into separate object ``LatticeBasis`` and use it as base class for ``Lattice``

### Code Refactoring
- use black code formatter

### Documentation
- add inheritance diagram to ``LatticeStructure`` and fix some docstrings
- add inheritance diagram to ``Lattice``
- add example to ``LatticeBasis`` docstring
- add attributes to docstring of ``LatticeBasis``
- improve docstring of ``Lattice`` class


<a name="0.6.7"></a>
## [0.6.7] - 2022-16-02
### New Features
- add method for hiding the box and axis of a plot
- Add ``finite_hypercubic`` lattice prefab
- use git-chglog to auto-generate changelogs

### Improvements/Bug Fixes
- add ``use_mplstyle`` to plotting module
- change atom parameter order and fix resulting errors
- use `box` for plot aspect ratio
- improve lattice plotting and fix scaling/auto-limit issues
- update change log template and include old entries

### Code Refactoring
- rename unitcell module to atom

### Documentation
- fix limits of plot in configuration tutorial
- update index page of docs
- fix docstrings of ``DataMap``
- add hamiltonian section to tutorial
- add change log contributing to documentation

<a name="0.6.6"></a>
## [0.6.6] - 2022-12-02

### Improved/Fixed

- improve build process
- improve periodic neighbor computation
- improve documentation
- improve CI/Tests
- minor fixes


<a name="0.6.5"></a>
## [0.6.5] - 2022-03-02

### New Features

- 2D/3D ``Shape`` object for easier lattice construction.
- repeat/extend built lattices.

### Improved/Fixed

- improve build process
- improve periodic neighbor computation (still not stable)
- add/improve tests
- improve plotting
- add more docstrings
- fix multiple bugs

[Unreleased]: https://github.com/dylanljones/lattpy/compare/0.7.3...HEAD
[0.7.3]: https://github.com/dylanljones/lattpy/compare/0.7.2...0.7.3
[0.7.2]: https://github.com/dylanljones/lattpy/compare/0.7.1...0.7.2
[0.7.1]: https://github.com/dylanljones/lattpy/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/dylanljones/lattpy/compare/0.6.7...0.7.0
[0.6.7]: https://github.com/dylanljones/lattpy/compare/0.6.6...0.6.7
[0.6.6]: https://github.com/dylanljones/lattpy/compare/0.6.5...0.6.6
[0.6.5]: https://github.com/dylanljones/lattpy/compare/0.6.4...0.6.5
[0.6.4]: https://github.com/dylanljones/lattpy/compare/0.6.3...0.6.4
[0.6.3]: https://github.com/dylanljones/lattpy/compare/0.6.2...0.6.3
[0.6.2]: https://github.com/dylanljones/lattpy/compare/0.6.1...0.6.2
[0.6.1]: https://github.com/dylanljones/lattpy/compare/0.6.0...0.6.1
