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


def main():
    fig, ax = plot_simple_square_periodic()
    fig.savefig("example_square_periodic.png")
    plt.show()


if __name__ == "__main__":
    main()
