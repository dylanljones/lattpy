# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones

import lattpy as lp
from lattpy.plotting import hide_box
import matplotlib.pyplot as plt


def plot_simple_square_periodic():
    latt = lp.simple_square()
    latt.build((5, 3), periodic=0)

    ax = latt.plot()
    fig = ax.get_figure()
    fig.tight_layout()
    return fig, ax


def plot_graphene_headerimage():
    latt = lp.graphene()
    latt.build((20.5, 6))
    ax = latt.plot(legend=False, lw=2, con_colors=[(0, 1, "0.3")])
    hide_box(ax, axis=True)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.set_size_inches(2 * 4.85, 2.5)

    return fig, ax


def main():
    # fig, ax = plot_simple_square_periodic()
    # fig.savefig("example_square_periodic.png")

    fig, ax = plot_graphene_headerimage()
    fig.tight_layout()
    fig.savefig("header.png", transparent=True, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
