"""
Implémentation des principales functions pour la méthode des éléments finis
en une dimension.

Cette implémentation se base sur le chapitre 12 des notes de cours
de David Sénéchal.
"""
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.integrate as integrate
from . import tools


class Grid:
    """Grid of points for the finite element method.

    :param points: List of sorted points of the grid.
    :type points: numpy.ndarray
    """

    def __init__(self, points):
        # On s'assure que les points sont ordonnées.
        if np.any((points[1:] - points[:-1]) < 0.0):
            raise ValueError("Points must be sorted.")
        self.points = points

    def __len__(self):
        """Number of points in the grid."""
        return len(self.points)

    def matrice_masse_interne(self):
        """Retourne la matrice de masse de la grille
        sous forme de scipy.sparse.csc_matrix.

        Les éléments aux frontières ne sont pas calculés.
        Ainsi, l'élément (0, 0) de la matrice retournée
        représente la valeur (1, 1) de la matrice complète.
        """
        n = len(self)
        matrice = sp.dok_matrix((n - 2, n - 2))
        # On calcule les valeurs sur la diagonale centrale.
        for site in range(1, n - 1):
            matrice[site - 1, site - 1] = (
                self.points[site + 1] - self.points[site - 1]
            ) / 3.0
        # On calcule les valeurs sur les autres diagonales.
        for site in range(1, n - 2):
            valeur = (self.points[site + 1] - self.points[site]) / 6.0
            matrice[site - 1, site] = valeur
            matrice[site, site - 1] = valeur
        return matrice.tocsc()

    def matrice_laplacienne_interne(self):
        """Retourne la matrice de l'opérateur différentiel (D^2) de la grille
        sous forme de scipy.sparse.csc_matrix.

        Les éléments aux frontières ne sont pas calculés.
        Ainsi, l'élément (0, 0) de la matrice retournée
        représente la valeur (1, 1) de la matrice complète.
        """
        n = len(self)
        matrice = sp.dok_matrix((n - 2, n - 2))
        # On calcule les valeurs sur la diagonale centrale.
        for site in range(1, n - 1):
            matrice[site - 1, site - 1] = sum(
                1.0 / (self.points[i] - self.points[i + 1])
                for i in [site - 1, site]
            )
        # On calcule les valeurs sur les autres diagonales.
        for site in range(1, n - 2):
            valeur = 1.0 / (self.points[site + 1] - self.points[site])
            matrice[site - 1, site] = valeur
            matrice[site, site - 1] = valeur
        return matrice.tocsc()

    def matrice_potentiel(self, potentiel):
        """Retourne la matrice de potentiel de la grille
        sous forme de scipy.sparse.csc_matrix.

        Les éléments aux frontières ne sont pas calculés.
        Ainsi, l'élément (0, 0) de la matrice retournée
        représente la valeur (1, 1) de la matrice complète.

        :param potentiel: Une fonction de float vers float représentant le potentiel à calculer.
        """
        # Calcule les ratios pour les intégrales.
        def pente_pos(x, site):
            num = x - self.points[site - 1]
            denom = self.points[site] - self.points[site - 1]
            return num / denom

        def pente_neg(x, site):
            num = self.points[site + 1] - x
            denom = self.points[site + 1] - self.points[site]
            return num / denom

        n = len(self)
        matrice = sp.dok_matrix((n - 2, n - 2))

        # On calcule les valeurs sur la diagonale centrale.
        for site in range(1, n - 1):
            valeur_pos = integrate.quad(
                lambda x: potentiel(x) * pente_pos(x, site)**2,
                self.points[site - 1],
                self.points[site]
            )[0]
            valeur_neg = integrate.quad(
                lambda x: potentiel(x) * pente_neg(x, site)**2,
                self.points[site],
                self.points[site + 1]
            )[0]
            matrice[site - 1, site - 1] = valeur_pos + valeur_neg

        # On calcule les valeurs sur les autres diagonales.
        for site in range(1, n - 2):
            valeur = integrate.quad(
                lambda x: (
                    potentiel(x)
                    * pente_pos(x, site + 1)
                    * pente_neg(x, site)
                ),
                self.points[site],
                self.points[site + 1]
            )[0]
            matrice[site - 1, site] = valeur
            matrice[site, site - 1] = valeur
        return matrice.tocsr()


@tools.time_it
def solve_shrodinger_using_fem(
        potential: callable, points: np.ndarray, n_states: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Function that solves the time independant shrodinger equation
    for a given 1D potential and returns the first "n_states" eigenstates
    of the hamiltonian. It calculates the wavefunction at every x=points
    using the finite element method.

    :param potential: The function that returns the potential at every x position.
    :type potential: Callable[[np.ndarray], np.ndarray]
    :param points: The x position of the grid points of shape (N).
    :type points: np.ndarray
    :param n_states: Number of eigenstates to find.
    :type n_states: int
    :return: The first "n_states" energies of the eigenstates and the first "n_states" eigenstates
        calculated at every grid points of shape (n_states, N).
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    raise NotImplementedError('You need to implement this function.')
