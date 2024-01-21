from typing import Callable, List, Tuple

import numpy as np
import scipy as sp
from scipy.optimize import brentq

from .constants import LEFT_BOUNDARY
from .tools import time_it, normalise, scope_roots


def schrodinger(
        state: np.ndarray,
        x: np.ndarray,
        potential: Callable[[np.ndarray], np.ndarray],
        energy: float,
) -> np.ndarray:
    r"""Calculates the Schrodinger equation for a given state.

    The Schrodinger equation is given by:

    .. math::
        \frac{d^2 \psi}{dx^2} = (V(x) - E) \psi

    where :math:`\psi` is the state, :math:`V(x)` is the potential, and :math:`E` is the energy.

    :param state: The state to calculate the Schrodinger equation for. The state is given by
        :math:`\psi` and :math:`\frac{d \psi}{dx}` in that order.
    :type state: np.ndarray
    :param x: The x values.
    :type x: np.ndarray
    :param potential: The potential to use in the Schrodinger equation.
    :type potential: Callable[[np.ndarray], np.ndarray]
    :param energy: The energy to use in the Schrodinger equation.
    :type energy: float
    :return: The Schrodinger output. The output is given by :math:`\frac{d^2 \psi}{dx^2}` and
        :math:`2(V(x) - E) \psi` in that order.
    :rtype: np.ndarray
    """
    psi, dpsi = state
    raise NotImplementedError("This function is not implemented yet.")


def get_right_boundary(energy: float, domain: np.ndarray, potential: callable) -> float:
    """Evaluates the numerical solution at given energy on the right bound of
    the domain.

    :param energy: The energy to use in the Schrodinger equation.
    :type energy: float
    :param domain: The x position of the grid points of shape (N).
    :type domain: np.ndarray
    :param potential: The function that returns the potential at every x position.
    :type potential: Callable[[np.ndarray], np.ndarray]
    :return: The value of the wavefunction evaluated at the rightmost boundary.
    :rtype: float
    """
    raise NotImplementedError("This function is not implemented yet.")


def solve_energies(energy0: float, domain: np.ndarray, potential: callable, sols_nb: int = 6) -> List[float]:
    """Uses an initial energy guess to find eigenenergies of the system using
    shooting method.

    :param energy0: Initial guess for fundamental energy.
    :type energy0: float
    :param domain: The x position of the grid points of shape (N).
    :type domain: np.ndarray
    :param potential: The function that returns the potential at every x position.
    :type potential: Callable[[np.ndarray], np.ndarray]
    :param sols_nb: Number of eigenenergies to find starting from 'energy0'.
    :type sols_nb: int
    :return: The eigenenergies.
    :rtype: list
    """
    raise NotImplementedError("This function is not implemented yet.")


@time_it
def solve_shrodinger_using_shooting(
        potential: callable, domain: np.ndarray, n_states: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Function that solves the time independant shrodinger equation
    for a given 1D potential and returns the first "n_states" eigenstates
    of the hamiltonian. It calculates the wavefunction at every x=points using
    the shooting method.

    :param potential: The function that returns the potential at every x position.
    :type potential: Callable[[np.ndarray], np.ndarray]
    :param domain: The x position of the grid points of shape (N).
    :type domain: np.ndarray
    :param n_states: Number of eigenstates to find.
    :type n_states: int
    :return: The first "n_states" energies of the eigenstates and the first "n_states" eigenstates
        calculated at every grid points of shape (n_states, N).
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    raise NotImplementedError("This function is not implemented yet.")

