import time
from typing import Tuple

import numpy as np
from functools import wraps


def time_it(func):
    """Wrapper that prints the processing time of a given function.
    """
    @wraps(func)
    def time_it_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.4f}.")

        return result

    return time_it_wrapper


def lin_extrapolate(xs, ys, xf):
    """ Function that makes a linear extrapolation of a point,
    given 2 reference points

    Parameters
    ----------
    xs: np.ndarray, shape=(2), the x position of the 2 reference points
    ys: np.ndarray, shape=(2), the y position of the 2 reference points
    xf: float, the y position of the point we wish to extrapolate

    Returns
    -------
    yf: float, the y value of the extrapolated point
    """
    a = (ys[0]-ys[1])/(xs[0]-xs[1])
    return ys[0] + a*(xf-xs[0])


def normalise(state, xs):
    """ Function that normalises a state vector assuming the vector is Real

    Parameters
    ----------
    state: np.ndarray, shape=(N), the state vector to normalise
    xs: np.ndarray, shape=(N), the x position of the 2 reference points

    Returns
    -------
    state_f: np.ndarray, shape=(N), normalised state
    """
    dxs = xs[1:] - xs[:-1]
    temp = state * state
    area = 1/2 * (temp[:-1] + temp[1:])*dxs
    area = area.sum()
    return state/np.sqrt(area)


def scope_roots(
        func: callable,
        x0: float,
        args: tuple = (),
        max_iters: int = 500
) -> Tuple[float, bool]:
    """Scope roots around point x=x0 using iterations of delta=1e-3.

    :param func: The function to scope roots for.
    :type func: callable
    :param x0: The point around which to scope roots.
    :type x0: float
    :param args: The arguments to pass to func.
    :type args: tuple
    :param max_iters: The maximum number of iterations to perform.
    :type max_iters: int
    :return: The root and whether the root was found.
    :rtype: tuple
    """
    raise NotImplementedError("This function is not implemented yet.")
