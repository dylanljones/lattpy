# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Contains plotting tools for the lattice and other related objects."""

import itertools
import numpy as np
from collections.abc import Iterable
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection, Collection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Line3D, Poly3DCollection
from matplotlib.artist import allow_rasterization
from matplotlib import path, transforms
import colorcet as cc

__all__ = [
    "subplot", "draw_line", "draw_lines", "hide_box",
    "draw_arrows", "draw_vectors", "draw_points", "draw_indices", "draw_unit_cell",
    "draw_surfaces", "interpolate_to_grid", "draw_sites"
]

# Golden ratio as standard ratio for plot-figures
GOLDEN_RATIO = (np.sqrt(5) - 1.0) / 2.0

# ======================================================================================
# Formatting / Styling
# ======================================================================================


def set_color_cycler(color_cycle=cc.glasbey_category10):
    """Sets the colors of the pyplot color cycler.

    Parameters
    ----------
    color_cycle : Sequence
        A list of the colors to use in the prop cycler.
    """
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_cycle)


def use_mplstyle(style, color_cycle=None):
    """Update matplotlib rcparams according to style.

    Parameters
    ----------
    style : str or dict or Path or Iterable
        The style configuration.
    color_cycle : Sequence, optional
        A list of the colors to use in the prop cycler.
    """
    mpl_style.use(style)
    if color_cycle is not None:
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_cycle)


def set_equal_aspect(ax=None, adjustable="box"):
    """Sets the aspect ratio of the plot to equal.

    Parameters
    ----------
    ax : Axes
        The axes of the plot. If not given the current axes is used.
    adjustable : None or {'box', 'datalim'}, optional
        If not None, this defines which parameter will be adjusted to meet
        the equal aspect ratio. If 'box', change the physical dimensions of
        the Axes. If 'datalim', change the x or y data limits.

    Notes
    -----
    Setting the aspect ratio to equal is not supported for 3D plots and will
    be ignored in that case.
    """
    if ax is None:
        ax = plt.gca()
    if ax.name == "3d":
        return
    ax.set_aspect("equal", adjustable)


def hide_box(ax, axis=False):
    """Remove the box and optionally the axis of a plot.

    Parameters
    ----------
    ax : Axes
        The axes to remove the box.
    axis : bool, optional
        If True the axis are hiden as well as the box.
    """
    if ax.name == "3d":
        return

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    if axis:
        for side in ["left", "bottom"]:
            ax.spines[side].set_visible(False)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])


# ======================================================================================
# General Plotting
# ======================================================================================

def subplot(dim, adjustable="box", ax=None):
    """Generates a two- or three-dimensional subplot with a equal aspect ratio

    Parameters
    ----------
    dim : int
        The dimension of the plot.
    adjustable : None or {'box', 'datalim'}, optional
        If not None, this defines which parameter will be adjusted to meet
        the equal aspect ratio. If 'box', change the physical dimensions of
        the Axes. If 'datalim', change the x or y data limits.
        Only applied to 2D plots.
    ax : Axes, optional
        Existing axes to format. If an existing axes is passed no new figure is created.

    Returns
    -------
    fig : Figure
        The figure of the subplot.
    ax : Axes
        The newly created or formatted axes of the subplot.
    """
    if dim > 3:
        raise ValueError(f"Plotting in {dim} dimensions is not supported!")
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)
    else:
        fig = ax.get_figure()
    set_equal_aspect(ax, adjustable)
    return fig, ax


# noinspection PyShadowingNames
def draw_line(ax, points, **kwargs):
    """Draw a line segment between multiple points.

    Parameters
    ----------
    ax : Axes
        The axes for drawing the line segment.
    points : (N, D) np.ndarray
        A list of points between the line is drawn.
    **kwargs
        Additional keyword arguments for drawing the line.

    Returns
    -------
    coll : Line2D or Line3D
        The created line.

    Examples
    --------
    >>> from lattpy import plotting
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> points = np.array([[1, 0], [0.7, 0.7], [0, 1], [-0.7, 0.7], [-1, 0]])
    >>> _ = plotting.draw_line(ax, points)
    >>> ax.margins(0.1, 0.1)
    >>> plt.show()

    """
    dim = len(points[0])
    if dim < 3:
        line = Line2D(*points.T, **kwargs)
    elif dim == 3:
        line = Line3D(*points.T, **kwargs)
    else:
        raise ValueError(f"Can't draw line with dimension {dim}")
    ax.add_line(line)
    return line


# noinspection PyShadowingNames
def draw_lines(ax, segments, **kwargs):
    """Draw multiple line segments between points.

    Parameters
    ----------
    ax : Axes
        The axes for drawing the lines.
    segments : array_like of (2, D) np.ndarray
        A list of point pairs between the lines are drawn.
    **kwargs
        Additional keyword arguments for drawing the lines.

    Returns
    -------
    coll: LineCollection or Line3DCollection
        The created line collection.

    Examples
    --------
    >>> from lattpy import plotting
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> segments = np.array([
    ...     [[0, 0], [1, 0]],
    ...     [[0, 1], [1, 1]],
    ...     [[0, 2], [1, 2]]
    ... ])
    >>> _ = plotting.draw_lines(ax, segments)
    >>> ax.margins(0.1, 0.1)
    >>> plt.show()
    """
    dim = len(segments[0][0])
    if dim < 3:
        coll = LineCollection(segments, **kwargs)
    elif dim == 3:
        coll = Line3DCollection(segments, **kwargs)
    else:
        raise ValueError(f"Can't draw lines with dimension {dim}")
    ax.add_collection(coll)
    return coll


# noinspection PyShadowingNames
def draw_vectors(ax, vectors, pos=None, **kwargs):
    """Draws multiple lines from an optional starting point in the given directions.

    Parameters
    ----------
    ax : Axes
        The axes for drawing the lines.
    vectors : (N, D) np.ndarray
        The vectors to draw.
    pos : (D, ) np.ndarray, optional
        The starting position of the vectors. The default is the origin.
    **kwargs
        Additional keyword arguments for drawing the lines.

    Returns
    -------
    coll: LineCollection or Line3DCollection
        The created line collection.

    Examples
    --------
    >>> from lattpy import plotting
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> vectors = np.array([[1, 0], [0.7, 0.7], [0, 1], [-0.7, 0.7], [-1, 0]])
    >>> _ = plotting.draw_vectors(ax, vectors, [1, 0])
    >>> ax.margins(0.1, 0.1)
    >>> plt.show()
    """
    pos = pos if pos is not None else np.zeros(len(vectors[0]))
    vectors = np.atleast_2d(vectors)
    # Fix 1D case
    if vectors.shape[1] == 1:
        vectors = np.hstack((vectors, np.zeros((vectors.shape[0], 1))))
        pos = np.array([pos[0], 0])
    # Build segments
    segments = list()
    for v in vectors:
        segments.append([pos, pos + v])
    return draw_lines(ax, segments, **kwargs)


# noinspection PyShadowingNames
def draw_arrows(ax, vectors, pos=None, **kwargs):
    """Draws multiple arrows from an optional starting point in the given directions.

    Parameters
    ----------
    ax : Axes
        The axes for drawing the arrows.
    vectors : (N, D) np.ndarray
        The vectors to draw.
    pos : (D, ) np.ndarray, optional
        The starting position of the vectors. The default is the origin.
    **kwargs
        Additional keyword arguments for drawing the arrows.

    Returns
    -------
    coll: LineCollection or Line3DCollection
        The created line collection.

    Examples
    --------
    >>> from lattpy import plotting
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> vectors = np.array([[1, 0], [0.7, 0.7], [0, 1], [-0.7, 0.7], [-1, 0]])
    >>> _ = plotting.draw_arrows(ax, vectors)
    >>> ax.margins(0.1, 0.1)
    >>> plt.show()
    """
    vectors = np.atleast_2d(vectors)
    num_vecs, dim = vectors.shape
    if pos is None:
        pos = np.zeros((num_vecs, dim))
    else:
        pos = np.atleast_2d(pos)
        if pos.shape[0] == 1:
            pos = np.tile(pos, (num_vecs, 1))
    assert len(pos) == len(vectors)
    points = pos.T
    directions = vectors.T
    end_points = (pos + vectors).T
    # Plot invisible points for datalim
    if dim == 1:
        end_points = np.append(end_points, np.zeros_like(end_points), axis=0)
        points = np.append(points, np.zeros_like(points), axis=0)
        directions = np.append(directions, np.zeros_like(directions), axis=0)
    ax.scatter(*end_points, s=0)
    # Draw arrows as quiver plot
    if dim != 3:
        kwargs.update({"angles": "xy", "scale_units": "xy", "scale": 1})
    else:
        kwargs.update({"normalize": False})
    return ax.quiver(*points, *directions, **kwargs)


# noinspection PyShadowingNames
def draw_points(ax, points, size=10, **kwargs):
    """Draws multiple points as scatter plot.

    Parameters
    ----------
    ax : Axes
        The axes for drawing the points.
    points : (N, D) np.ndarray
        The positions of the points to draw.
    size : float, optional
        The size of the markers of the points.
    **kwargs
        Additional keyword arguments for drawing the points.

    Returns
    -------
    scat : PathCollection
        The scatter plot item.

    Examples
    --------
    >>> from lattpy import plotting
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> points = np.array([[1, 0], [0.7, 0.7], [0, 1], [-0.7, 0.7], [-1, 0]])
    >>> _ = plotting.draw_points(ax, points)
    >>> ax.margins(0.1, 0.1)
    >>> plt.show()
    """
    points = np.atleast_2d(points)
    # Fix 1D case
    if points.shape[1] == 1:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))

    scat = ax.scatter(*points.T, s=size**2, **kwargs)
    # Manualy update data-limits
    # ax.ignore_existing_data_limits = True
    datalim = scat.get_datalim(ax.transData)
    ax.update_datalim(datalim)
    return scat


# noinspection PyShadowingNames
def draw_surfaces(ax, vertices, **kwargs):
    """Draws a 3D surfaces defined by a set of vertices.

    Parameters
    ----------
    ax : Axes3D
        The axes for drawing the surface.
    vertices : array_like
        The vertices defining the surface.
    **kwargs
        Additional keyword arguments for drawing the lines.

    Returns
    -------
    surf : Poly3DCollection
        The surface object.

    Examples
    --------
    >>> from lattpy import plotting
    >>> import matplotlib.pyplot as plt
    >>> vertices = [[0, 0, 0], [1, 1, 0], [0.5, 0.5, 1]]
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection="3d")
    >>> _ = plotting.draw_surfaces(ax, vertices, alpha=0.5)
    >>> plt.show()
    """
    if not isinstance(vertices[0][0], Iterable):
        vertices = [list(vertices)]
    poly = Poly3DCollection(vertices, **kwargs)
    ax.add_collection3d(poly)
    return poly


# noinspection PyShadowingNames
def text(ax, strings, positions, offset=None, **kwargs):
    """Adds multiple strings to a plot.

    Parameters
    ----------
    ax : Axes
        The axes for drawing the text.
    strings : str or sequence of str
        The text to render.
    positions : (..., D) array_like
        The positions of the texts.
    offset : float or (D, ) array_like
        The offset of the positions of the text.
    **kwargs
        Additional keyword arguments for drawing the text.

    Returns
    -------
    texts : list
        The text items.

    Examples
    --------
    >>> from lattpy import plotting
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> points = np.array([[-1, 0], [-0.7, 0.7], [0, 1], [0.7, 0.7], [1, 0]])
    >>> strings = ["A", "B", "C", "D", "E"]
    >>> _ = plotting.text(ax, strings, points)
    >>> _ = ax.set_xlim(-1.5, +1.5)
    >>> _ = ax.set_ylim(-0.5, +1.5)
    >>> plt.show()
    """
    positions = np.atleast_2d(positions)
    texts = list()
    if offset is None:
        offset = np.zeros(max(2, len(positions[0])))
    elif isinstance(offset, float):
        offset = offset * np.ones(max(2, len(positions[0])))

    for s, pos in zip(strings, positions):
        if len(pos) == 1:
            pos = [pos, 0]
        tpos = np.asarray(pos) + offset
        txt = ax.text(*tpos, s=s, **kwargs)
        texts.append(txt)
    return texts


# ======================================================================================
# Lattice plotting
# ======================================================================================

# noinspection PyAbstractClass
class CircleCollection(Collection):
    """Custom circle collection

    The default matplotlib `CircleCollection` creates circles based on their
    area in screen units. This class uses the radius in data units. It behaves
    like a much faster version of a `PatchCollection` of `Circle`.
    The implementation is similar to `EllipseCollection`.
    """
    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)
        self.radius = np.atleast_1d(radius)
        self._paths = [path.Path.unit_circle()]
        self.set_transform(transforms.IdentityTransform())
        self._transforms = np.empty((0, 3, 3))

    def _set_transforms(self):
        ax = self.axes
        self._transforms = np.zeros((self.radius.size, 3, 3))
        self._transforms[:, 0, 0] = self.radius * ax.bbox.width / ax.viewLim.width
        self._transforms[:, 1, 1] = self.radius * ax.bbox.height / ax.viewLim.height
        self._transforms[:, 2, 2] = 1

    @allow_rasterization
    def draw(self, renderer):
        self._set_transforms()
        super().draw(renderer)


# noinspection PyShadowingNames
def draw_sites(ax, points, radius=0.2, **kwargs):
    """Draws multiple circles with a scaled radius.

    Parameters
    ----------
    ax : Axes
        The axes for drawing the points.
    points : (N, D) np.ndarray
        The positions of the points to draw.
    radius : float
        The radius of the points. Scaling is only supported for 2D plots!
    **kwargs
        Additional keyword arguments for drawing the points.

    Returns
    -------
    point_coll : CircleCollection or PathCollection
        The circle or path collection.


    Examples
    --------
    >>> from lattpy import plotting
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> points = np.array([[1, 0], [0.7, 0.7], [0, 1], [-0.7, 0.7], [-1, 0]])
    >>> _ = plotting.draw_sites(ax, points, radius=0.2)
    >>> _ = ax.set_xlim(-1.5, +1.5)
    >>> _ = ax.set_ylim(-0.5, +1.5)
    >>> plotting.set_equal_aspect(ax)
    >>> plt.show()
    """
    points = np.atleast_2d(points)
    # Fix 1D case
    if points.shape[1] == 1:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))

    dim = points.shape[1]
    if dim < 3:
        col = CircleCollection(radius, offsets=points, transOffset=ax.transData,
                               **kwargs)
        ax.add_collection(col)
        label = kwargs.get("label", "")
        if label:
            ax.plot([], [], marker='o', lw=0, color=kwargs.get("color", None),
                    label=label, markersize=10)
        datalim = col.get_datalim(ax.transData)
        datalim.x0 -= radius
        datalim.x1 += radius
        datalim.y0 -= radius
        datalim.y1 += radius
        ax.update_datalim(datalim)
        return col
    else:
        size = radius * 50
        scat = ax.scatter(*points.T, s=size**2, **kwargs)
        # Manualy update data-limits
        # ax.ignore_existing_data_limits = True
        datalim = scat.get_datalim(ax.transData)
        ax.update_datalim(datalim)
        return scat


# noinspection PyShadowingNames
def draw_indices(ax, positions, offset=0.05, **kwargs):
    """Draws the indices of the given positions on the plot.

    Parameters
    ----------
    ax : Axes
        The axes for drawing the text.
    positions : (..., D) array_like
        The positions of the texts.
    offset : float or (D, ) array_like
        The offset of the positions of the texts.
    **kwargs
        Additional keyword arguments for drawing the text.

    Returns
    -------
    texts : list
        The text items.

    Examples
    --------
    >>> from lattpy import plotting
    >>> import matplotlib.pyplot as plt
    >>> points = np.array([[-1, 0], [-0.7, 0.7], [0, 1], [0.7, 0.7], [1, 0]])
    >>> fig, ax = plt.subplots()
    >>> _ = plotting.draw_points(ax, points)
    >>> _ = plotting.draw_indices(ax, points)
    >>> ax.margins(0.1, 0.1)
    >>> plt.show()
    """
    strings = [str(i) for i in range(len(positions))]
    va = "bottom"
    ha = "left"
    return text(ax, strings, positions, offset, ha=ha, va=va, **kwargs)


# noinspection PyShadowingNames
def draw_unit_cell(ax, vectors, outlines=True, **kwargs):
    """Draws the basis vectors and unit cell.

    Parameters
    ----------
    ax : Axes
        The axes for drawing the text.
    vectors : float or (D, D) array_like
        The vectors defining the basis.
    outlines : bool, optional
        If True the box define dby the basis vectors (unit cell) is drawn.
    **kwargs
        Additional keyword arguments for drawing the lines.

    Returns
    -------
    lines : list
        A list of the plotted lines.

    Examples
    --------
    >>> from lattpy import plotting
    >>> import matplotlib.pyplot as plt
    >>> vectors = np.array([[1, 0], [0, 1]])
    >>> fig, ax = plt.subplots()
    >>> _ = plotting.draw_unit_cell(ax, vectors)
    >>> plt.show()
    """
    dim = len(vectors)
    color = kwargs.pop("color", "k")

    arrows = draw_arrows(ax, vectors, color=color, **kwargs)
    lines = list()
    if outlines and dim > 1:
        for v, pos in itertools.permutations(vectors, r=2):
            data = np.asarray([pos, pos + v]).T
            line = ax.plot(*data, color=color, **kwargs)[0]
            lines.append(line)
        if dim == 3:
            for vecs in itertools.permutations(vectors, r=3):
                v, pos = vecs[0], np.sum(vecs[1:], axis=0)
                data = np.asarray([pos, pos + v]).T
                line = ax.plot(*data, color=color, **kwargs)[0]
                lines.append(line)
    return arrows, lines


def interpolate_to_grid(positions, values, num=(100, 100), offset=(0., 0.),
                        method="linear", fill_value=np.nan):
    x, y = positions.T

    # Create regular grid
    xi = np.linspace(min(x) - offset[0], max(x) + offset[0], num[0])
    yi = np.linspace(min(y) - offset[1], max(y) + offset[1], num[1])

    # Interpolate data to grid
    xx, yy = np.meshgrid(xi, yi)
    zz = griddata((x, y), values, (xi[None, :], yi[:, None]), method, fill_value)
    return xx, yy, zz
