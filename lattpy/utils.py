# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import time
import numpy as np
from typing import Iterable, List, Sequence, Optional, Union
import logging

logging.captureWarnings(True)


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
        super().__init__("lattice doesn't contain any atoms",
                         "use 'add_atom' to add an 'Atom'-object")


class NoBaseNeighboursError(ConfigurationError):

    def __init__(self):
        msg = "base neighbours not configured"
        hint = "call 'calculate_distances' after adding atoms or " \
               "use the 'neighbours' keyword of 'add_atom'"
        super().__init__(msg, hint)


class NotBuiltError(ConfigurationError):

    def __init__(self):
        msg = "lattice has not been built"
        hint = "use the 'build' method to construct a finite size lattice model"
        super().__init__(msg, hint)


def split_index(index):
    return index[:-1], index[-1]


def interweave(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """ Interweaves multiple arrays along the first axis

    Example
    -------
    >>> arr1 = np.array([[1, 1], [3, 3], [5, 5]])
    >>> arr2 = np.array([[2, 2], [4, 4], [6, 6]])
    >>> interweave([arr1, arr2])
    array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])

    Parameters
    ----------
    arrays: (M) Sequence of (N, ...) array_like
        The input arrays to interwave. The shape of all arrays must match.

    Returns
    -------
    interweaved: (M*N, ....) np.ndarray
    """
    shape = list(arrays[0].shape)
    shape[0] = sum(x.shape[0] for x in arrays)
    result = np.empty(shape, dtype=arrays[0].dtype)
    n = len(arrays)
    for i, arr in enumerate(arrays):
        result[i::n] = arr
    return result


def vindices(limits: Iterable[Sequence[int]], sort_axis: Optional[int] = 0,
             dtype: Optional[Union[int, str, np.dtype]] = None) -> np.ndarray:
    """ Return an array representing the indices of a d-dimensional grid.

    Parameters
    ----------
    limits: (D, 2) array_like
        The limits of the indices for each axis.
    sort_axis: int, optional
        Optional axis that is used to sort indices.
    dtype: int or str or np.dtype, optional
        Optional data-type for storing the lattice indices. By default the given limits
        are checked to determine the smallest possible data-type.

    Returns
    -------
    vectors: (N, D) np.ndarray
    """
    if dtype is None:
        # Estimate needed data type. Use the negative of the maximal
        # absolute value to force data type to be signed
        dtype = str(np.min_scalar_type(-np.max(np.abs(limits))))
        dtype = dtype[1:] if dtype.startswith("u") else dtype
    limits = np.asarray(limits)
    dim = limits.shape[0]

    # Create meshgrid reshape grid to array of indices

    # version 1:
    # axis = np.meshgrid(*(np.arange(*lim, dtype=dtype) for lim in limits))
    # nvecs = np.asarray([np.asarray(a).flatten("F") for a in axis]).T

    # version 2:
    # slices = [slice(lim[0], lim[1], 1) for lim in limits]
    # nvecs = np.mgrid[slices].astype(dtype).reshape(dim, -1).T

    # version 3:
    size = limits[:, 1] - limits[:, 0]
    nvecs = np.indices(size, dtype=dtype).reshape(dim, -1).T + limits[:, 0]

    # Optionally sort indices along given axis
    if sort_axis is not None:
        nvecs = nvecs[np.lexsort(nvecs.T[[sort_axis]])]

    return nvecs


def vrange(start=None, *args,
           dtype: Optional[Union[int, str, np.dtype]] = None,
           sort_axis: Optional[int] = 0, **kwargs) -> np.ndarray:
    """ Return evenly spaced vectors within a given interval.

    Parameters
    ----------
    start: array_like, optional
        The starting value of the interval. The interval includes this value.
        The default start value is 0.
    stop: array_like
        The end value of the interval.
    step: array_like, optional
        Spacing between values. If `start` and `stop` are sequences and the `step`
        is a scalar the given step size is used for all dimensions of the vectors.
        The default step size is 1.
    sort_axis: int, optional
        Optional axis that is used to sort indices.
    dtype: dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    Returns
    -------
    vectors: (N, D) np.ndarray
    """
    # parse arguments
    if len(args) == 0:
        stop = start
        start = np.zeros_like(stop)
        step = kwargs.get("step", 1.0)
    elif len(args) == 1:
        stop = args[0]
        step = kwargs.get("step", 1.0)
    else:
        stop, step = args

    start = np.atleast_1d(start)
    stop = np.atleast_1d(stop)
    if step is None:
        step = np.ones_like(start)
    elif not hasattr(step, "__len__"):
        step = np.ones_like(start) * step

    # Create grid and reshape to array of vectors
    slices = [slice(i, f, s) for i, f, s in zip(start, stop, step)]
    array = np.mgrid[slices].reshape(len(slices), -1).T
    # Optionally sort array along given axis
    if sort_axis is not None:
        array = array[np.lexsort(array.T[[sort_axis]])]

    return array if dtype is None else array.astype(dtype)


def chain(items: Sequence, cycle: bool = False) -> List:
    """Create chain between items

    Parameters
    ----------
    items: array_like
        items to join to chain
    cycle: bool, optional
        cycle to the start of the chain if True, default: False

    Returns
    -------
    chain: list
        chain of items

    Example
    -------
    >>> print(chain(["x", "y", "z"]))
    [['x', 'y'], ['y', 'z']]

    >>> print(chain(["x", "y", "z"], True))
    [['x', 'y'], ['y', 'z'], ['z', 'x']]
    """
    result = list()
    for i in range(len(items)-1):
        result.append([items[i], items[i+1]])
    if cycle:
        result.append([items[-1], items[0]])
    return result


def frmt_num(num, dec=1, unit='', div=1000.) -> str:
    """Returns a formatted string of a numbe

    Parameters
    ----------
    num: float
        The number to format.
    dec: int
        Number of decimals.
    unit: str, optional
        Optional unit suffix.
    div: float, optional
        The divider used for units. The default is `1000`.

    Returns
    -------
    num_str: str
    """
    for prefix in ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < div:
            return f"{num:.{dec}f}{prefix}{unit}"
        num /= div
    return f"{num:.{dec}f}Y{unit}"


def frmt_time(seconds: float, short: bool = False, width: int = 0) -> str:
    """Returns a formated string for a given time in seconds.

    Parameters
    ----------
    seconds: float
        Time value to format
    short: bool, optional
        Flag if short representation should be used.
    width: int, optional
        Optional minimum length of the returned string

    Returns
    -------
    time_str: str
    """
    string = "00:00"

    # short time string
    if short:
        if seconds > 0:
            mins, secs = divmod(seconds, 60)
            if mins > 60:
                hours, mins = divmod(mins, 60)
                string = f"{hours:02.0f}:{mins:02.0f}h"
            else:
                string = f"{mins:02.0f}:{secs:02.0f}"

    # Full time strings
    else:
        if seconds < 1e-3:
            nanos = 1e6 * seconds
            string = f"{nanos:.0f}\u03BCs"
        elif seconds < 1:
            millis = 1000 * seconds
            string = f"{millis:.1f}ms"
        elif seconds < 60:
            string = f"{seconds:.1f}s"
        else:
            mins, seconds = divmod(seconds, 60)
            if mins < 60:
                string = f"{mins:.0f}:{seconds:04.1f}min"
            else:
                hours, mins = divmod(mins, 60)
                string = f"{hours:.0f}:{mins:02.0f}:{seconds:02.0f}h"

    if width > 0:
        string = f"{string:>{width}}"
    return string


class Timer:

    __slots__ = ["_time", "_t0"]

    def __init__(self, method=time.perf_counter):
        self._time = method
        self._t0 = 0
        self.start()

    @property
    def seconds(self):
        return self.time() - self._t0

    @property
    def millis(self):
        return 1000 * (self.time() - self._t0)

    def time(self):
        return self._time()

    def start(self):
        self._t0 = self._time()

    def eta(self, progress: float) -> float:
        if not progress:
            return 0.0
        else:
            return (1 / progress - 1) * self.time()

    def strfrmt(self, short: bool = False, width: int = 0) -> str:
        return frmt_time(self.seconds, short, width)

    @staticmethod
    def sleep(t):
        time.sleep(t)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.strfrmt(short=True)})'

    def __str__(self) -> str:
        return self.strfrmt(short=True)
