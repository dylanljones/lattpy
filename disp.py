# coding: utf-8
"""
Created on 20 Apr 2020
author: Dylan Jones
"""
import numpy as np
import matplotlib.pyplot as plt
from lattpy.utils import vlinspace, chain
from lattpy import simple_chain, simple_square, simple_cubic


def brillouin_zone(latt):
    v_rec = latt.reciprocal_vectors()
    qlims = list()
    for d in range(latt.dim):
        b = v_rec[d]
        dlim = -b/2, +b/2
        qlims.append(dlim)
    return qlims


def get_dispersion_segments(points, n=100):
    sections = list()
    for p1, p2 in chain(points, cycle=True):
        sections.append(vlinspace(p1, p2, n=n))
    return sections


def dispersion(q, d=1.0, m=1.0):
    omega2 = 4*d/m * (1 - np.cos(q))
    return np.sqrt(omega2)


def phi_matrix(delta, d=1):
    if np.linalg.norm(delta) == 0:
        return 4 * d * np.eye(2)
    else:
        phi = np.zeros((2, 2))
        phi[1, 0] = phi[0, 1] = -d
        return phi


def dynamic_matrix(q, deltas):
    dmat = np.zeros((2, 2))
    for delta in deltas:
        dmat += phi_matrix(delta) * np.exp(1j * np.dot(q, delta)).real
    return dmat


def main():
    latt = simple_square()
    qlims = brillouin_zone(latt)

    points = [0.0, 0.0], [np.pi, 0.0], [np.pi, np.pi]
    segments = get_dispersion_segments(points)

    deltas = latt.get_neighbour_vectors(0, include_zero=True)
    disp = list()
    for qvecs in segments:
        for q in qvecs:
            dmat = dynamic_matrix(q, deltas)
            omega = np.linalg.eigvals(dmat)
            disp.append(omega)

    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(disp)
    plt.show()


if __name__ == "__main__":
    main()
