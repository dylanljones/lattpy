# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from lattpy.unitcell import Atom


def test_atom_uniqueness():
    atom1 = Atom("A")
    atom2 = Atom("A")
    assert atom1 == atom2
    assert atom1.__hash__() == atom2.__hash__()

    atom1 = Atom("A")
    atom2 = Atom("B")
    assert atom1 != atom2
    assert atom1.__hash__() != atom2.__hash__()

    atom1 = Atom()
    atom2 = Atom()
    assert atom1 != atom2
    assert atom1.__hash__() != atom2.__hash__()


def test_atom_params():
    atom = Atom("A", color="r", energy=1)

    assert atom.color == "r"
    assert atom["color"] == "r"

    assert atom.energy == 1
    assert atom["energy"] == 1

    atom["spin"] = 1
    assert atom.spin == 1
    assert atom["spin"] == 1

    atom.spin = 1
    assert atom.spin == 1
    assert atom["spin"] == 1

    assert atom.get("spin") == 1

    del atom["spin"]

    assert atom.get("spin", None) is None


def test_atom_copy():
    atom = Atom("A", energy=1.0)

    copy = atom.copy()
    assert copy.name == atom.name
    assert copy.energy == atom.energy

    atom.energy = 2.0

    assert copy.energy != atom.energy
