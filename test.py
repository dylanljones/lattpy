# coding: utf-8
"""
Created on 08 Apr 2020
author: Dylan Jones
"""
import numpy as np
import lattpy as lp
from lattpy import Atom
import matplotlib.pyplot as plt
from lattpy.geometrie import Plane, get_center


def compute_plains(latt, idx=12):
    neighbours = latt.neighbours(12, range(latt.n_dist), unique=False)
    print(neighbours)

    plot = latt.plot(show=False, show_indices=True)
    p0 = latt.position(idx)
    planes = list()
    for i in neighbours:
        p1 = latt.position(i)
        center = get_center(p0, p1)
        vec = p1 - p0
        plane = Plane(center, vec)
        pp = plane.perp()
        planes.append(pp)

    for i, p1 in enumerate(planes):
        for j, p2 in enumerate(planes[:i]):
            inter = p1.intersection(p2)
            print(inter)
            if not np.any(inter == np.nan):
                print(inter)
    for p in planes:
        print(p.x, p.v)
        plot.draw_line(p.x-p.v, p.x+p.v, color='r', lw=2)

    plot.show()


def main():
    shape = (10, 6)
    atom = Atom('C', color='k', size=7)

    latt = lp.graphene()
    latt.build(shape)

    latt.plot()


if __name__ == "__main__":
    main()
