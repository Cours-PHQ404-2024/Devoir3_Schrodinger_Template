import numpy as np
import pytest
import pickle
import os
from src import fem
from src import shooting


def sqr_potential(x):
    return x * x/2


potential_funcs = {func.__name__: func for func in [sqr_potential]}


test_data = {}
for function in ["solve_shrodinger_using_fem", "solve_shrodinger_using_shooting"]:
    test_data[function] = pickle.load(
        open(os.path.join(os.path.dirname(__file__), "test_data", f"{function}.pkl"), "rb")
    )


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (inputs_expected_dict["inputs"], inputs_expected_dict["output"])
        for inputs_expected_dict in test_data["solve_shrodinger_using_fem"]
    ],
)
def test_solve_shrodinger_using_fem(inputs, expected):
    inputs = (potential_funcs[inputs[0]], *inputs[1:])
    pred = fem.solve_shrodinger_using_fem(*inputs)
    for p, t in zip(pred, expected):
        np.testing.assert_allclose(p, t, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (inputs_expected_dict["inputs"], inputs_expected_dict["output"])
        for inputs_expected_dict in test_data["solve_shrodinger_using_shooting"]
    ],
)
def test_solve_shrodinger_using_shooting(inputs, expected):
    inputs = (potential_funcs[inputs[0]], *inputs[1:])
    pred = shooting.solve_shrodinger_using_shooting(*inputs)
    for p, t in zip(pred, expected):
        np.testing.assert_allclose(p, t, atol=1e-4, rtol=1e-4)

