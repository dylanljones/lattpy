# coding: utf-8
#
# This code is part of pylattice.
# 
# Copyright (c) 2020, Dylan Jones


class LatticeError(Exception):
    pass


class ConfigurationError(LatticeError):

    def __init__(self, msg="", hint=""):
        super().__init__(msg, hint)

    @property
    def msg(self):
        return self.args[0]

    @property
    def hint(self):
        return self.args[1]

    def __str__(self):
        msg, hint = self.args
        if hint:
            msg += f" ({hint})"
        return msg


class SiteOccupiedError(ConfigurationError):

    def __init__(self, atom, pos):
        super().__init__(f"Can't add {atom} to lattice, position {pos} already occupied!")


class NoAtomsError(ConfigurationError):

    def __init__(self):
        super().__init__("lattice doesn't contain any atoms", "use 'add_atom' to add an 'Atom'-object")


class NoBaseNeighboursError(ConfigurationError):

    def __init__(self):
        msg = "base neighbours not configured"
        hint = "call 'calculate_distances' after adding atoms or use the 'neighbours' keyword of 'add_atom'"
        super().__init__(msg, hint)


class NotBuiltError(ConfigurationError):

    def __init__(self):
        msg = "lattice has not been built"
        hint = "use the 'build' method to construct a finite size lattice model"
        super().__init__(msg, hint)
