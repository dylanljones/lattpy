# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Contains miscellaneous utility methods."""

import logging
from typing import Iterable, List, Sequence, Union, Tuple
import numpy as np

__all__ = [
    "ArrayLike", "logger", "LatticeError", "ConfigurationError", "SiteOccupiedError",
    "NoAtomsError", "NoConnectionsError", "NotAnalyzedError", "NotBuiltError",
    "min_dtype", "chain", "create_lookup_table", "frmt_num", "frmt_bytes", "frmt_time",
]

# define type for numpy `array_like` types
ArrayLike = Union[int, float, Iterable, np.ndarray]


# Configure package logger
logger = logging.getLogger("lattpy")

_CH = logging.StreamHandler()
_CH.setLevel(logging.DEBUG)

_FRMT_STR = "[%(asctime)s] %(levelname)-8s - %(name)-15s - %(message)s"
_FRMT = logging.Formatter(_FRMT_STR, datefmt='%H:%M:%S')

_CH.setFormatter(_FRMT)             # Add formatter to stream handler
logger.addHandler(_CH)              # Add stream handler to package logger

logger.setLevel(logging.WARNING)    # Set initial logging level


class LatticeError(Exception):

    pass


class ConfigurationError(LatticeError):

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
        super().__init__(f"Can't add {atom} to lattice, "
                         f"position {pos} already occupied!")


class NoAtomsError(ConfigurationError):

    def __init__(self):
        super().__init__("lattice doesn't contain any atoms",
                         "use 'add_atom' to add an 'Atom'-object")


class NotAnalyzedError(ConfigurationError):

    def __init__(self):
        msg = "lattice not analyzed"
        hint = "call 'analyze' after adding atoms and connections or " \
               "use the 'analyze' keyword of 'add_connection'"
        super().__init__(msg, hint)


class NoConnectionsError(ConfigurationError):

    def __init__(self):
        msg = "base neighbors not configured"
        hint = "call 'add_connection' after adding atoms or " \
               "use the 'neighbors' keyword of 'add_atom'"
        super().__init__(msg, hint)


class NotBuiltError(ConfigurationError):

    def __init__(self):
        msg = "lattice has not been built"
        hint = "use the 'build' method to construct a finite size lattice model"
        super().__init__(msg, hint)


def create_lookup_table(array: ArrayLike,
                        dtype: Union[str, np.dtype] = np.uint8
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Converts the given array to an array of indices linked to the unique values.

    Parameters
    ----------
    array : array_like
    dtype : int or np.dtype, optional
            Optional data-type for storing the indices of the unique values.
            By default `np.uint8` is used, since it is assumed that the
            input-array has only a few unique values.

    Returns
    -------
    values : np.ndarray
        The unique values occuring in the input-array.
    indices : np.ndarray
        The corresponding indices in the same shape as the input-array.
    """
    values = np.sort(np.unique(array))
    indices = np.zeros_like(array, dtype=dtype)
    for i, x in enumerate(values):
        mask = array == x
        indices[mask] = i
    return values, indices


def min_dtype(a: Union[int, float, np.ndarray, Iterable],
              signed: bool = True) -> np.dtype:
    """Returns the minimum required dtype to store the given values.

    Parameters
    ----------
    a : array_like
        One or more values for determining the dtype.
        Should contain the maximal expected values.
    signed : bool, optional
        If `True` the dtype is forced to be signed. The default is `True`.

    Returns
    -------
    dtype : dtype
        The required dtype.
    """
    if signed:
        a = -np.max(np.abs(a)) - 1
    else:
        amin, amax = np.min(a), np.max(a)
        if amin < 0:
            a = - amax - 1 if abs(amin) <= amax else amin
        else:
            a = amax
    return np.dtype(np.min_scalar_type(a))


def chain(items: Sequence, cycle: bool = False) -> List:
    """Creates a chain between items

    Parameters
    ----------
    items : Sequence
        items to join to chain
    cycle : bool, optional
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
    for i in range(len(items) - 1):
        result.append([items[i], items[i + 1]])
    if cycle:
        result.append([items[-1], items[0]])
    return result


def frmt_num(num: float, dec: int = 1, unit: str = "",
             div: float = 1000.) -> str:
    """Returns a formatted string of a number.

    Parameters
    ----------
    num : float
        The number to format.
    dec : int, optional
        Number of decimals. The default is 1.
    unit : str, optional
        Optional unit suffix. By default no unit-strinmg is used.
    div : float, optional
        The divider used for units. The default is 1000.

    Returns
    -------
    num_str: str
    """
    for prefix in ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < div:
            return f"{num:.{dec}f}{prefix}{unit}"
        num /= div
    return f"{num:.{dec}f}Y{unit}"  # pragma: no cover


def frmt_bytes(num: float, dec: int = 1) -> str:  # pragma: no cover
    """Returns a formatted string of the number of bytes."""
    return frmt_num(num, dec, unit="iB", div=1024)


def frmt_time(seconds: float, short: bool = False, width: int = 0):  # pragma: no cover
    """Returns a formated string for a given time in seconds.

    Parameters
    ----------
    seconds : float
        Time value to format
    short : bool, optional
        Flag if short representation should be used.
    width : int, optional
        Optional minimum length of the returned string.

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
