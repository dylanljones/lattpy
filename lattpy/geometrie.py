# coding: utf-8
"""
Created on 21 Apr 2020
author: Dylan Jones
"""
import numpy as np


def get_center(r1, r2):
    vec = r2 - r1
    return r1 + vec/2


class Plane:

    def __init__(self, x, v):
        self.x = x
        self.v = v
        self.dim = len(x)

    def get(self, lamb=1.0):
        return self.x + lamb * self.v

    def perp(self, x=None):
        if self.dim == 2:
            v_perp = np.array([-self.v[1], self.v[0]])
        else:
            if self.v[1] == 0 and self.v[2] == 0:
                v_perp = np.cross(self.v, [0, 1, 0])
            else:
                v_perp = np.cross(self.v, [1, 0, 0])

        if x is None:
            x = self.x.copy()
        return Plane(x, v_perp)

    def parallel(self, x=None):
        if x is None:
            x = self.x.copy()
        return Plane(x, self.v.copy())

    def intersection(self, other):
        return (self.x - other.x) / (other.v - self.v)


def main():
    pass


if __name__ == "__main__":
    main()
