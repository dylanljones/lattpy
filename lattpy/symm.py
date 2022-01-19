# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

# flake8: noqa

"""Symmetry operations and analyzers.

References
----------
https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/symmetry/analyzer.py

"""

import logging
import itertools
import numpy as np
import scipy.cluster as spcluster
from collections import defaultdict

logger = logging.getLogger(__name__)


# =========================================================================
# SYMMETRY OPERATIONS
# =========================================================================


def affine_matrix(rot=None, trans=None):
    if rot is None:
        rot = np.eye(3)
    if trans is None:
        trans = np.zeros(3)
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = trans
    return mat


def affine_from_axis_angle_trans(axis, angle: float, trans=None, degree: bool = True):
    axis = np.array(axis)
    trans = np.array(trans) if trans is not None else np.zeros(3)

    angle = np.deg2rad(angle) if degree else angle
    cosa = np.cos(angle)
    sina = np.sin(angle)
    u = axis / np.linalg.norm(axis)

    rot = np.zeros((3, 3))
    rot[0, 0] = cosa + u[0] ** 2 * (1 - cosa)
    rot[0, 1] = u[0] * u[1] * (1 - cosa) - u[2] * sina
    rot[0, 2] = u[0] * u[2] * (1 - cosa) + u[1] * sina
    rot[1, 0] = u[0] * u[1] * (1 - cosa) + u[2] * sina
    rot[1, 1] = cosa + u[1] ** 2 * (1 - cosa)
    rot[1, 2] = u[1] * u[2] * (1 - cosa) - u[0] * sina
    rot[2, 0] = u[0] * u[2] * (1 - cosa) - u[1] * sina
    rot[2, 1] = u[1] * u[2] * (1 - cosa) + u[0] * sina
    rot[2, 2] = cosa + u[2] ** 2 * (1 - cosa)

    return affine_matrix(rot, trans)


def affine_from_axis_angle_origin(
        axis, angle: float, origin=None, degree: bool = True):
    if origin is None:
        origin = np.zeros(3)
    theta = np.deg2rad(angle) if degree else angle
    a, b, c = origin
    u, v, w = axis
    # Set some intermediate values.
    u2 = u * u
    v2 = v * v
    w2 = w * w
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    l2 = u2 + v2 + w2
    lsqrt = np.sqrt(l2)

    # Build the matrix entries element by element.
    m11 = (u2 + (v2 + w2) * cos_t) / l2
    m12 = (u * v * (1 - cos_t) - w * lsqrt * sin_t) / l2
    m13 = (u * w * (1 - cos_t) + v * lsqrt * sin_t) / l2
    m14 = (
              a * (v2 + w2)
              - u * (b * v + c * w)
              + (u * (b * v + c * w) - a * (v2 + w2)) * cos_t
              + (b * w - c * v) * lsqrt * sin_t
          ) / l2

    m21 = (u * v * (1 - cos_t) + w * lsqrt * sin_t) / l2
    m22 = (v2 + (u2 + w2) * cos_t) / l2
    m23 = (v * w * (1 - cos_t) - u * lsqrt * sin_t) / l2
    m24 = (
              b * (u2 + w2)
              - v * (a * u + c * w)
              + (v * (a * u + c * w) - b * (u2 + w2)) * cos_t
              + (c * u - a * w) * lsqrt * sin_t
          ) / l2

    m31 = (u * w * (1 - cos_t) - v * lsqrt * sin_t) / l2
    m32 = (v * w * (1 - cos_t) + u * lsqrt * sin_t) / l2
    m33 = (w2 + (u2 + v2) * cos_t) / l2
    m34 = (
              c * (u2 + v2)
              - w * (a * u + b * v)
              + (w * (a * u + b * v) - c * (u2 + v2)) * cos_t
              + (a * v - b * u) * lsqrt * sin_t
          ) / l2

    mat = [
        [m11, m12, m13, m14],
        [m21, m22, m23, m24],
        [m31, m32, m33, m34],
        [0, 0, 0, 1],
    ]
    return np.array(mat)


def affine_inversion(origin=None):
    if origin is None:
        origin = np.zeros(3)
    mat = -np.eye(4)
    mat[3, 3] = 1
    mat[0:3, 3] = 2 * np.array(origin)
    return mat


def affine_reflection(normal, origin=None):
    if origin is None:
        origin = np.zeros(3)
    # Normalize the normal vector first.
    n = np.array(normal, dtype=float) / np.linalg.norm(normal)

    u, v, w = n

    translation = np.eye(4)
    translation[0:3, 3] = -np.array(origin)

    xx = 1 - 2 * u ** 2
    yy = 1 - 2 * v ** 2
    zz = 1 - 2 * w ** 2
    xy = -2 * u * v
    xz = -2 * u * w
    yz = -2 * v * w
    mirror_mat = [[xx, xy, xz, 0], [xy, yy, yz, 0], [xz, yz, zz, 0], [0, 0, 0, 1]]

    if np.linalg.norm(origin) > 1e-6:
        mirror_mat = np.dot(np.linalg.inv(translation), np.dot(mirror_mat, translation))
    return mirror_mat


def affine_rotoreflection(axis, angle, origin=None, degree: bool = True):
    if origin is None:
        origin = np.zeros(3)
    rot = affine_from_axis_angle_origin(axis, angle, origin, degree)
    ref = affine_reflection(axis, origin)
    return np.dot(rot, ref)


class SymmetryOperation:

    def __init__(self, affine_mat, tol=None, name=""):
        if tol is None:
            tol = 0.01
        self.mat = np.array(affine_mat)
        self.name = name
        self.tol = tol

    @property
    def rotation_mat(self):
        return self.mat[:3, :3]

    @property
    def translation_vec(self):
        return self.mat[:3, 3]

    def apply(self, points):
        points = np.array(points)

        # Ensure list of points even for single point
        single = len(points.shape) == 1
        points = np.atleast_2d(points)

        # Cast to 3d
        dim = points.shape[1]
        if points.shape[1] < 3:
            points3d = np.zeros((points.shape[0], 3))
            points3d[:, :dim] = points
            points = points3d

        # Apply transformation
        affine_points = np.concatenate([points, np.ones(points.shape[:-1] + (1,))],
                                       axis=-1)
        res = np.inner(affine_points, self.mat)[..., :-1]

        # cast back to original dim
        if dim < 3:
            res = res[:, :dim]
        # return single array if input was a point
        if single:
            res = res[0]

        return res

    def __hash__(self):
        return hash(str(self))

    def __call__(self, points):
        return self.apply(points)

    def __eq__(self, other):
        return np.allclose(self.mat, other.mat, atol=self.tol)

    def __mul__(self, other):
        """Combines two symmetry operations in one."""
        new_matrix = np.dot(self.mat, other.affine_matrix)
        return SymmetryOperation(new_matrix)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __str__(self):
        return str(self.mat)

    @classmethod
    def eye(cls, tol=None):
        return cls(np.eye(4), tol, "Identity")

    @classmethod
    def rot_trans(cls, rot=None, trans=None, tol=None, name="C"):
        mat = affine_matrix(rot, trans)
        return cls(mat, tol, name)

    @classmethod
    def rot_axis_trans(cls, axis, angle: float, trans=None, degree: bool = True,
                       tol=None, name="C"):
        mat = affine_from_axis_angle_trans(axis, angle, trans, degree)
        return cls(mat, tol, name)

    @classmethod
    def rot_axis_origin(cls, axis, angle: float, origin=None, degree: bool = True,
                        tol=None, name="C"):
        mat = affine_from_axis_angle_origin(axis, angle, origin, degree)
        return cls(mat, tol, name)

    @classmethod
    def inversion(cls, origin=None, tol=None):
        mat = affine_inversion(origin)
        return cls(mat, tol, "Inversion")

    @classmethod
    def reflection(cls, normal, origin=None, tol=None, name="σ"):
        mat = affine_reflection(normal, origin)
        return cls(mat, tol, name)

    @classmethod
    def rotoreflection(cls, axis, angle, origin=None, degree: bool = True,
                       tol: float = None, name="S"):
        mat = affine_rotoreflection(axis, angle, origin, degree)
        return cls(mat, tol, name)


# =========================================================================
# POINT GROUP ANALYZER
# =========================================================================


def find_point(p, points, atol=1e-8):
    if len(points) == 0:
        return []
    diff = np.array(points) - np.array(p)[None, :]
    return np.where(np.all(np.abs(diff) < atol, axis=1))[0]


def is_equivalent(positions, atoms, transformed, tol=1e-4):
    for i, pos in enumerate(transformed):
        atom = atoms[i]
        ind = find_point(pos, positions, tol)
        if not (len(ind) == 1 and atoms[ind[0]].is_identical(atom)):
            return False
    return True


def is_valid_symop(positions, atoms, symop, tol=1e-4):
    for i, pos in enumerate(positions):
        atom = atoms[i]
        ind = find_point(symop.apply(pos), positions, tol)
        if not (len(ind) == 1 and atoms[ind[0]].is_identical(atom)):
            return False
    return True


def find_unique_symops(positions, atoms, symops, tol=1e-4):
    unique_symops = dict()
    for symop in symops:
        transformed = symop.apply(positions)

        # Shift center of mass to origin
        weights = np.array([atom.weight for atom in atoms])
        mc = np.average(transformed, weights=weights, axis=0)
        transformed -= mc

        # Check for equivalent configs
        for other_symop, other_pos in unique_symops.items():
            if is_equivalent(transformed, atoms, other_pos, tol):
                logger.debug("%s is equivalent to %s", symop.__repr__(),
                             other_symop.__repr__())
                break
        else:
            logger.debug("Found unique symop %s", symop.__repr__())
            unique_symops[symop] = transformed

    return unique_symops


def cluster_sites(positions, atoms, tol):
    """Cluster sites based on distance and atom type.

    Parameters
    ----------
    positions : array_like
    atoms : list of Atom
    tol : float

    Returns
    -------
    origin : int or None
        Index of site at the center of mass. None if there are no origin atoms.
    clustered : dict
        Dictionary of clustered sites with format {(avg_dist, species_and_occu):
        [list of sites]}.
    """
    num_atoms = len(positions)
    # Cluster works for dim > 2 data. We just add a dummy 0 for second coordinate.
    dists = [[np.linalg.norm(pos), 0] for pos in positions]

    f = spcluster.hierarchy.fclusterdata(dists, tol, criterion="distance")
    clustered_dists = defaultdict(list)
    for i in range(num_atoms):
        clustered_dists[f[i]].append(dists[i])
    avg_dist = {label: np.mean(val) for label, val in clustered_dists.items()}
    clustered_sites = defaultdict(list)
    origin_site = None
    for i, atom in enumerate(atoms):
        if avg_dist[f[i]] < tol:
            origin_site = i
        else:
            clustered_sites[(avg_dist[f[i]], atom.name)].append(i)
    return origin_site, clustered_sites


class PointGroupAnalyzer:

    inversion_op = SymmetryOperation.inversion()

    def __init__(self, *args, tol=0.3, eig_tol=1e-2, mat_tol=1e-1):
        if len(args) == 1:
            mol = args[0]
            positions = mol.positions3d
            atoms = mol.atoms
        else:
            positions, atoms = args

        self._positions = positions
        self._atoms = atoms
        self._num_sites = len(self.positions)
        self._tol = tol
        self._eig_tol = eig_tol
        self._mat_tol = mat_tol

        self.principal_axes = None
        self.eigvals = None
        self.rot_sym = []
        self.sym_ops = []
        self.sch_symbol = ""

        self._analyze()

    @property
    def positions(self):
        return self._positions

    @property
    def num_sites(self):
        return self._num_sites

    @property
    def atoms(self):
        return self._atoms

    def _analyze(self) -> None:
        if len(self.positions) == 1:
            self.sch_symbol = "Kh"
        else:
            inertia_tensor = np.zeros((3, 3))
            total_inertia = 0
            for alpha in range(self.num_sites):
                pos = self.positions[alpha]
                atom = self.atoms[alpha]
                wt = atom.weight
                for i in range(3):
                    inertia_tensor[i, i] += wt * (pos[(i + 1) % 3] ** 2 +
                                                  pos[(i + 2) % 3] ** 2)
                for i, j in [(0, 1), (1, 2), (0, 2)]:
                    inertia_tensor[i, j] += -wt * pos[i] * pos[j]
                    inertia_tensor[j, i] += -wt * pos[j] * pos[i]
                total_inertia += wt * np.dot(pos, pos)
            # Normalize the inertia tensor so that it does not scale with size
            # of the system.  This mitigates the problem of choosing a proper
            # comparison tolerance for the eigenvalues.
            inertia_tensor /= total_inertia
            eigvals, eigvecs = np.linalg.eig(inertia_tensor)
            self.principal_axes = eigvecs.T
            self.eigvals = eigvals

            v1, v2, v3 = eigvals
            tol = self._eig_tol

            self.rot_sym = []
            self.sym_ops = [SymmetryOperation.eye()]
            if abs(v1 * v2 * v3) < tol:  # Eig zero
                logger.debug("Linear molecule detected")
                self._linear()
            elif abs(v1 - v2) < tol and abs(v1 - v3) < tol:  # All same
                logger.debug("Spherical top molecule detected")
                self._spherical_top()
            elif abs(v1 - v2) > tol and abs(v1 - v3) > tol and abs(v2 - v3) > tol:
                # All different
                logger.debug("Asymmetric top molecule detected")
                self._asymmetric_top()
            else:
                logger.debug("Symmetric top molecule detected")
                self._symmetric_top()

    def is_valid_symop(self, symop: SymmetryOperation) -> bool:
        """Check if a particular symmetry operation is a valid operation for a molecule.

        A valid symmetry operations maps all atoms to another equivalent atom.

        Parameters
        ----------
        symop : SymmetryOperation
            Symmetry operation to test.

        Returns
        -------
        valid : bool
            Whether the operation is a valid symmetry operation for a molecule.
        """
        return is_valid_symop(self.positions, self.atoms, symop, self._tol)

    def add_valid_symop(self, symop: SymmetryOperation) -> bool:
        """Check if a particular symmetry operation is valid and adds it if so.

        A valid symmetry operations maps all atoms to another equivalent atom.

        Parameters
        ----------
        symop : SymmetryOperation
            Symmetry operation to test.

        Returns
        -------
        valid : bool
            Whether the operation is a valid symmetry operation for a molecule.
        """
        is_valid = self.is_valid_symop(symop)
        if is_valid:
            self.sym_ops.append(symop)
        return is_valid

    def _check_2fold_axes_asym(self) -> None:
        """Test for 2-fold rotation along the principal axes."""
        for v in self.principal_axes:
            op = SymmetryOperation.rot_axis_trans(v, 180, name="C2")
            if self.is_valid_symop(op):
                self.sym_ops.append(op)
                self.rot_sym.append((v, 2))

    def _find_mirror(self, axis) -> str:
        """Looks for a mirror symmetry of specified type about axis.

        Possible types are "h" or "vd".  Horizontal (h) mirrors are perpendicular to
        the axis while vertical (v) or diagonal (d) mirrors are parallel.
        "v" mirrors have atoms lying on the mirror plane while "d" mirrors do not.
        """
        mirror_type = ""

        # First test whether the axis itself is the normal to a mirror plane.
        symop = SymmetryOperation.reflection(axis, name="σh")
        if self.is_valid_symop(symop):
            self.sym_ops.append(symop)
            mirror_type = "h"
        else:
            # Iterate through all pairs of atoms to find mirror
            for i, j in itertools.combinations(range(self.num_sites), 2):
                pos1 = self.positions[i]
                pos2 = self.positions[j]
                name1 = self.atoms[i].name
                name2 = self.atoms[j].name
                if name1 == name2:
                    normal = pos1 - pos2
                    if np.dot(normal, axis) < self._tol:
                        op = SymmetryOperation.reflection(normal)
                        if self.is_valid_symop(op):
                            self.sym_ops.append(op)
                            if len(self.rot_sym) > 1:
                                mirror_type = "d"
                                for v, r in self.rot_sym:
                                    if np.linalg.norm(v - axis) >= self._tol:
                                        if np.dot(v, normal) < self._tol:
                                            mirror_type = "v"
                                            break
                            else:
                                mirror_type = "v"
                            op.name = "σ" + mirror_type
                            break
        return mirror_type

    def _find_spherical_axes(self):
        """Looks for R5, R4, R3 and R2 axes in spherical top molecules.

        Point group T molecules have only one unique 3-fold and one unique 2-fold axis.
        O molecules have one unique 4, 3 and 2-fold axes. I molecules have a unique
        5-fold axis.
        """
        rot_present = defaultdict(bool)

        origin_site, dist_el_sites = cluster_sites(self.positions, self.atoms,
                                                   self._tol)

        test_set = min(dist_el_sites.values(), key=lambda s: len(s))
        coords = [self.positions[i] for i in test_set]
        for c1, c2, c3 in itertools.combinations(coords, 3):
            for cc1, cc2 in itertools.combinations([c1, c2, c3], 2):
                if not rot_present[2]:
                    test_axis = cc1 + cc2
                    if np.linalg.norm(test_axis) > self._tol:
                        op = SymmetryOperation.rot_axis_trans(test_axis, 180, name="C2")
                        rot_present[2] = self.is_valid_symop(op)
                        if rot_present[2]:
                            self.sym_ops.append(op)
                            self.rot_sym.append((test_axis, 2))

            test_axis = np.cross(c2 - c1, c3 - c1)
            if np.linalg.norm(test_axis) > self._tol:
                for r in (3, 4, 5):
                    if not rot_present[r]:
                        op = SymmetryOperation.rot_axis_trans(test_axis, 360 / r,
                                                              name=f"C{r}")
                        rot_present[r] = self.is_valid_symop(op)
                        if rot_present[r]:
                            self.sym_ops.append(op)
                            self.rot_sym.append((test_axis, r))
                            break
            if rot_present[2] and rot_present[3] and (rot_present[4] or rot_present[5]):
                break

    def _get_smallest_set_not_on_axis(self, axis):
        """Returns the smallest list of atoms with the same species and
        distance from origin AND does not lie on the specified axis.

        This maximal set limits the possible rotational symmetry operations, since atoms
        lying on a test axis is irrelevant in testing rotational symmetry operations.
        """

        def not_on_axis(site):
            v = np.cross(self.positions[site], axis)
            return np.linalg.norm(v) > self._tol

        valid_sets = []
        origin_site, dist_el_sites = cluster_sites(self.positions, self.atoms,
                                                   self._tol)
        for test_set in dist_el_sites.values():
            valid_set = list(filter(not_on_axis, test_set))
            if len(valid_set) > 0:
                valid_sets.append(valid_set)

        return min(valid_sets, key=lambda s: len(s))

    def _check_rot_sym(self, axis):
        """Determines the rotational symmetry about supplied axis.

        Used only for symmetric top molecules which has possible rotational symmetry
        operations > 2.
        """
        min_set = self._get_smallest_set_not_on_axis(axis)
        max_sym = len(min_set)
        for i in range(max_sym, 0, -1):
            if max_sym % i != 0:
                continue
            op = SymmetryOperation.rot_axis_trans(axis, 360 / i, name=f"C{i}")
            rotvalid = self.is_valid_symop(op)
            if rotvalid:
                self.sym_ops.append(op)
                self.rot_sym.append((axis, i))
                return i
        return 1

    def _check_perpendicular_r2_axis(self, axis):
        """Checks for R2 axes perpendicular to unique axis.

        For handling symmetric top molecules.
        """
        min_set = self._get_smallest_set_not_on_axis(axis)
        for s1, s2 in itertools.combinations(min_set, 2):
            pos1 = self.positions[s1]
            pos2 = self.positions[s2]
            test_axis = np.cross(pos1 - pos2, axis)
            if np.linalg.norm(test_axis) > self._tol:
                op = SymmetryOperation.rot_axis_trans(test_axis, 180, name="C2")
                r2present = self.is_valid_symop(op)
                if r2present:
                    self.sym_ops.append(op)
                    self.rot_sym.append((test_axis, 2))
                    return True
        return None

    # =========================================================================

    def _linear(self) -> None:
        """Handles linear molecules."""
        if self.is_valid_symop(self.inversion_op):
            self.sym_ops.append(self.inversion_op)
            self.sch_symbol = "D*h"
        else:
            self.sch_symbol = "C*v"

    def _spherical_top(self) -> None:
        """Handles spherical top molecules.

        Belong to the T, O or I point groups.
        """

        self._find_spherical_axes()
        if len(self.rot_sym) == 0:
            logger.debug("Accidental speherical top!")
            self._symmetric_top()
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        if rot < 3:
            logger.debug("Accidental speherical top!")
            self._symmetric_top()
        elif rot == 3:
            mirror_type = self._find_mirror(main_axis)
            if mirror_type != "":
                if self.is_valid_symop(self.inversion_op):
                    self.sym_ops.append(self.inversion_op)
                    self.sch_symbol = "Th"
                else:
                    self.sch_symbol = "Td"
            else:
                self.sch_symbol = "T"
        elif rot == 4:
            if self.is_valid_symop(self.inversion_op):
                self.sym_ops.append(self.inversion_op)
                self.sch_symbol = "Oh"
            else:
                self.sch_symbol = "O"
        elif rot == 5:
            if self.is_valid_symop(self.inversion_op):
                self.sym_ops.append(self.inversion_op)
                self.sch_symbol = "Ih"
            else:
                self.sch_symbol = "I"

    def _no_rot_sym(self):
        """Handles molecules without any rotational symmetries.

        Only possible point groups are C1, Cs and Ci.
        """
        self.sch_symbol = "C1"
        if self.is_valid_symop(self.inversion_op):
            self.sch_symbol = "Ci"
            self.sym_ops.append(self.inversion_op)
        else:
            for v in self.principal_axes:
                mirror_type = self._find_mirror(v)
                if not mirror_type == "":
                    self.sch_symbol = "Cs"
                    break

    def _dihedral(self):
        """Handles dihedral molecules.

        Have intersecting R2 axes and a main axis.
        """
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        self.sch_symbol = f"D{rot}"
        mirror_type = self._find_mirror(main_axis)
        if mirror_type == "h":
            self.sch_symbol += "h"
        elif not mirror_type == "":
            self.sch_symbol += "d"

    def _cyclic(self):
        """Handles cyclic molecules."""
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        self.sch_symbol = f"C{rot}"
        mirror_type = self._find_mirror(main_axis)
        if mirror_type == "h":
            self.sch_symbol += "h"
        elif mirror_type == "v":
            self.sch_symbol += "v"
        elif mirror_type == "":
            op = SymmetryOperation.rotoreflection(main_axis, angle=180 / rot,
                                                  name=f"S{rot}")
            if self.is_valid_symop(op):
                # self.sym_ops.append(op)
                self.sch_symbol = f"S{2 * rot}"

    def _asymmetric_top(self) -> None:
        """Handles asymmetric top molecules

        Cannot contain rotational symmetries larger than 2-fold.
        """
        self._check_2fold_axes_asym()

        if len(self.rot_sym) == 0:
            logger.debug("No rotation symmetries detected.")
            self._no_rot_sym()
        elif len(self.rot_sym) == 3:
            logger.debug("Dihedral group detected.")
            self._dihedral()
        else:
            logger.debug("Cyclic group detected.")
            self._cyclic()

    def _symmetric_top(self) -> None:
        """Handles symmetric top molecules.

        Has one unique eigenvalue whose corresponding principal axis is a unique
        rotational axis. More complex handling required to look for R2 axes
        perpendicular to this unique axis.
        """
        if abs(self.eigvals[0] - self.eigvals[1]) < self._eig_tol:
            ind = 2
        elif abs(self.eigvals[1] - self.eigvals[2]) < self._eig_tol:
            ind = 0
        else:
            ind = 1
        logger.debug("Eigenvalues = %s." % self.eigvals)
        unique_axis = self.principal_axes[ind]
        self._check_rot_sym(unique_axis)
        logger.debug("Rotation symmetries = %s" % self.rot_sym)
        if len(self.rot_sym) > 0:
            self._check_perpendicular_r2_axis(unique_axis)

        if len(self.rot_sym) >= 2:
            self._dihedral()
        elif len(self.rot_sym) == 1:
            self._cyclic()
        else:
            self._no_rot_sym()
