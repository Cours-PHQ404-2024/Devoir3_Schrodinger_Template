import pytest
import numpy as np
from src.fem import Grid


@pytest.mark.parametrize(
    "points,expected", [
        (
                np.arange(5), np.array([[2 / 3, 1 / 6, 0],
                                        [1 / 6, 2 / 3, 1 / 6],
                                        [0, 1 / 6, 2 / 3]])
        )
    ]
)
def test_matrice_masse_interne(points, expected):
    grille = Grid(points)
    np.testing.assert_allclose(
        grille.matrice_masse_interne().toarray(),
        expected
    )


@pytest.mark.parametrize(
    "points,expected", [
        (
                np.arange(5), np.array([[-2,  1,   0],
                                        [ 1,  -2,  1],
                                        [ 0,   1, -2]])
        )
    ]
)
def test_matrice_laplacienne_interne(points, expected):
    grille = Grid(points)
    np.testing.assert_allclose(
        grille.matrice_laplacienne_interne().toarray(),
        expected
    )


@pytest.mark.parametrize(
    "points,potential_func,expected", [
        (
                np.arange(5),
                lambda x: x**2,
                np.array([[0.733, 0.383,    0],
                         [0.383, 2.733,  1.05],
                         [    0,  1.05, 6.067]])
        )
    ]
)
def test_matrice_potentiel(points, potential_func, expected):
    grille = Grid(points)
    np.testing.assert_allclose(
        grille.matrice_potentiel(potential_func).toarray(),
        expected,
        atol=1e-3,
    )
