from pytest import *
from logistic import *
import numpy as np
import numpy.testing as npt
from math import isclose


@fixture
def context():

    c = np.matrix([[0, 1, 0, 1, 1], [1, 0, 1, 0, 0]])

    y = np.matrix(
        [
            [0.66880, 0.73012, 0.77439, 0.35863, 0.075],
            [0.25940, 0.2475, 0.29837, 0.71791, 0.47666],
            [0.89621, 0.2170, 0.41113, 0.88443, 0.70431],
        ]
    )
    x = np.matrix([[0.16560, 0.31011, 0.80083], [0.48993, 0.64718, 0.34954]])

    V = np.matrix(
        [[0.69445055, 0.28388688, 0.65259926], [0.8014543, 0.07353407, 0.11769408]]
    )

    yield locals().copy()


def test_softmax(context):
    """
    n = 5, nc = 2, nf = 3
    """

    c = context["c"]
    y = context["y"]
    x = context["x"]
    assert isclose(softmax(c, y, x), 0.69405447)


def test_softmax_gradient(context):
    c = context["c"]
    y = context["y"]
    x = context["x"]

    npt.assert_array_almost_equal(
        softmax_gradient(c, y, x),
        np.array(
            [
                [0.01801635, -0.08771875, -0.04469136],
                [-0.01801635, 0.08771875, 0.04469136],
            ]
        ),
    )


def test_steepest_descent(context):
    c = context["c"]
    y = context["y"]
    x = context["x"]

    result = steepest_descent(c, y, x)
    npt.assert_array_almost_equal(
        np.array(result),
        np.array([[-0.46269, 1.632227, 0.704434], [1.11822, -0.674937, 0.445936]]),
    )


def test_hessian_sub(context):
    c = context["c"]
    y = context["y"]
    x = context["x"]
    V = context["V"]

    sub = hessian_sub(y, x, V)
    print(sub)
    npt.assert_array_almost_equal(
        sub,
        np.array(
            [
                [0.19301198, 0.21093543, 0.34022797],
                [-0.19301198, -0.21093543, -0.34022797],
            ]
        ),
    )


def test_cgls(context):
    c = context["c"]
    y = context["y"]
    x = context["x"]
    V = context["V"]
    hessian_handle = lambda z: hessian_sub(y, x, z)
    d = cgls(hessian_handle, V)

    diff = V - hessian_sub(y, x, d)
    print("quality", np.linalg.norm(diff))
    npt.assert_array_almost_equal(
        d, np.array([[-0.327545, -0.218315, 0.633983], [0.327545, 0.218315, -0.633983]])
    )
