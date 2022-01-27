# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones

import lattpy as lp
import matplotlib.pyplot as plt


def plot_simple_square_periodic():
    latt = lp.simple_square()
    latt.build((5, 3), periodic=0)

    ax = latt.plot()
    fig = ax.get_figure()
    fig.tight_layout()
    return fig, ax


def plot_simple_rectangular():
    latt = lp.Lattice([[2, 0], [0, 1]])
    latt.add_atom()
    # Two distances are needed for all connections
    latt.add_connections(2)
    latt.build((5, 3))

    ax = latt.plot()
    fig = ax.get_figure()
    fig.tight_layout()
    return fig, ax


def plot_square_two_atoms():
    latt = lp.Lattice.square()
    latt.add_atom([0.0, 0.0], "A")
    latt.add_atom([0.5, 0.5], "B")
    latt.add_connection("A", "A", 1)
    latt.add_connection("A", "B", 1)
    latt.analyze()
    latt.build((5, 3))

    ax = latt.plot()
    fig = ax.get_figure()
    fig.tight_layout()
    return fig, ax


def main():
    fig, ax = plot_simple_square_periodic()
    fig.savefig("example_square_periodic.png")

    fig, ax = plot_simple_rectangular()
    fig.savefig("example_rectangular.png")

    fig, ax = plot_square_two_atoms()
    fig.savefig("example_square_two_atoms.png")

    plt.show()


if __name__ == "__main__":
    main()
