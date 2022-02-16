# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from lattpy.atom import Atom


def test_atom_uniqueness():
    atom1 = Atom("A")
    atom2 = Atom("A")
    assert atom1 == atom2
    assert atom1.__hash__() == atom2.__hash__()

    atom1 = Atom("A")
    atom2 = Atom("B")
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


def test_atom_to_dict():
    atom = Atom("A", energy=1.0)
    expected = {"name": "A", "color": None, "radius": 0.2, "energy": 1.0}
    actual = atom.dict()
    actual.pop("index")
    assert actual == expected


def test_atom_param_length():
    atom = Atom("A", energy=1.0)
    assert len(atom) == 3


def test_atom_iter():
    atom = Atom("A", energy=1.0)
    assert list(atom) == ["color", "radius", "energy"]


def test_atoms_equal():
    atom1 = Atom("A", energy=1.0)
    atom2 = Atom("B")
    atom3 = Atom("B", energy=1.0)
    assert atom1 != atom2
    assert atom2 == atom3
    assert atom1 == "A"
    assert atom1 != "B"
    assert atom2 == "B"
    assert atom3 == "B"
