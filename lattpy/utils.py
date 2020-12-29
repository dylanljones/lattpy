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


def vrange(axis_ranges: Iterable, sort_axis=0) -> List:
    """ Return evenly spaced vectors within a given interval.

    Parameters
    ----------
    axis_ranges: array_like
        ranges for each axis.
    sort_axis: int, optional
        Optional axis that is used to sort vectors.

    Returns
    -------
    vectors: list
    """
    axis = np.meshgrid(*axis_ranges)
    nvecs = np.asarray([np.asarray(a).flatten("F") for a in axis]).T
    nvecs = nvecs[np.lexsort(nvecs.T[[sort_axis]])]
    return nvecs


def vlinspace(start: Union[float, Sequence[float]],
              stop: Union[float, Sequence[float]],
              n: Optional[int] = 1000) -> np.ndarray:
    """ Vector linspace

    Parameters
    ----------
    start: array_like or float
        d-dimensional start-point
    stop: array_like or float
        d-dimensional stop-point
    n: int, optional
        number of points, default=1000

    Returns
    -------
    vectors: np.ndarray
    """
    start = np.atleast_1d(start)
    stop = np.atleast_1d(stop)
    if not hasattr(start, '__len__') and not hasattr(stop, '__len__'):
        return np.linspace(start, stop, n)
    axes = [np.linspace(start[i], stop[i], n) for i in range(len(start))]
    return np.asarray(axes).T


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
