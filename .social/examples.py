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


def plot_graphene_periodic():
    latt = lp.graphene()
    latt.build((20.7, 10.5))
    ax = latt.plot(legend=False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig = ax.get_figure()
    fig.tight_layout()
    fig.set_size_inches(4.85, 2.5)

    return fig, ax


def main():
    # fig, ax = plot_simple_square_periodic()
    # fig.savefig("example_square_periodic.png")

    fig, ax = plot_graphene_periodic()
    fig.tight_layout()
    fig.savefig("example_graphene_periodic.png", transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
