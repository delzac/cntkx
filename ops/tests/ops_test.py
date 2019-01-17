import cntk as C
from cntkx.ops import cumsum, hardmax, erf
import numpy as np
from numpy.testing import assert_equal


def test_cumsum():
    a = C.input_variable(5)
    b = cumsum(a)

    n = np.array([1, 2, 3, 4, 5]).astype(np.float32)[None, ...]
    results = b.eval({a: n})
    assert_equal(results[0], n.cumsum())


def test_hardmax():
    a = C.input_variable((3, 5))

    n = np.array([[0.2, 0.2, 0.3, 0.2, 0.2],
                  [0.2, 0.4, 0.3, 0.2, 0.2],
                  [0.5, 0.2, 0.3, 0.2, 0.2]]).astype(np.float32)[None, ...]

    m = np.array([[0, 0, 1, 0, 0],
                  [0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0]])

    results = hardmax(a).eval({a: n})

    assert_equal(results[0], m)

    n = np.array([[0.2, 0.2, 0.3, 0.2, 0.2],
                  [0.2, 0.4, 0.3, 0.2, 0.2],
                  [0.5, 0.2, 0.3, 0.2, 0.2]]).astype(np.float32)[None, ...]

    m = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0]])

    results = hardmax(a, axis=None).eval({a: n})

    assert_equal(results[0], m)


def test_erf():
    a = C.input_variable(1)
    b = erf(a)

    n = np.array([[0], [0.02], [0.04], [0.06], [0.08], [0.1]], dtype=np.float32)
    ans = np.array([[0], [0.022564575], [0.045111106], [0.067621594], [0.090078126], [0.112462916]])
    results = b.eval({a: n})

    np.testing.assert_almost_equal(np.array(results), ans)

    a = C.input_variable(6)
    b = erf(a)

    n = np.array([[0, 0.02, 0.04, 0.06, 0.08, 0.1],], dtype=np.float32)
    ans = np.array([[0, 0.022564575, 0.045111106, 0.067621594, 0.090078126, 0.112462916],])
    results = b.eval({a: n})

    np.testing.assert_almost_equal(np.array(results), ans)
