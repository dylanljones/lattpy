# coding: utf-8
"""
Created on 12 May 2020
author: Dylan Jones
"""
import itertools


class Atom:

    COUNTER = itertools.count()

    def __init__(self, name=None, color=None, size=10, **kwargs):
        idx = next(self.COUNTER)
        name = name or str(idx)
        self.name = name
        self.col = color
        self.size = size
        self.kwargs = kwargs

    def __getitem__(self, item):
        return self.kwargs[item]

    def label(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Atom):
            return self.name == other.name
        else:
            return self.name == other

    def __repr__(self):
        return f"Atom({self.name}, {self.col}, {self.size})"

