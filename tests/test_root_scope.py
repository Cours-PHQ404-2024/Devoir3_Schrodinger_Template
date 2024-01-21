import numpy as np
import pytest
from numpy.testing import assert_array_compare

from src import tools
import operator


def quadratic(x: float) -> float:
    """Generic quadratic function.
    """
    return x**2 - 1


@pytest.mark.parametrize(
    "func,x0", [
        (quadratic, x0)
        for x0 in np.linspace(-1, 1, 10)
    ]
)
def test_scope_root(func, x0):
    """Root scoping test for quadratic equation.
    """
    y0_sign = np.sign(func(x0))
    x1_pred = tools.scope_roots(func, x0)[0]
    y1_sign = np.sign(func(x1_pred))
    assert_array_compare(operator.__ne__, y0_sign, y1_sign)
