from cntkx.ops.random import sample_top_k, sample
import numpy as np
import cntk as C


def test_sample():
    a = C.input_variable(5)
    b = sample(a, axis=-1)

    n = np.array([[0.1, 0.1, 0.1, 0.1, 100],] * 100)

    result = b.eval({a: n})

    np.testing.assert_equal(np.sum(result, axis=0), np.array([0, 0, 0, 0, 100]))

    a = C.input_variable(2)
    b = sample(a, axis=-1)

    n = np.array([[0.1, 0.1],] * 10000)
    result = b.eval({a: n})

    np.testing.assert_almost_equal(np.sum(result, axis=0), np.array([5000, 5000]), decimal=-2)

    a = C.input_variable((10, 2))
    b = sample(a, axis=-1)

    assert b.shape == (10, 2)

    n = np.ones((10000, 10, 2))
    result = b.eval({a: n})

    actual = np.sum(result, axis=0)
    desired = np.sum(n, axis=0) / 2
    np.testing.assert_almost_equal(actual, desired, decimal=-2)


def test_top_k_sample():
    a = C.input_variable(5)
    b = sample_top_k(a, k=3, num_classes=5)

    n = np.array([[1, 2, 3, 4, 5],] * 1000)
    assert n.ndim == 2
    assert n.shape == (1000, 5)

    results = b.eval({a: n})
    assert np.sum(results[:, :2]) == 0
    assert np.sum(results[:, 2:]) == 1000

    a = C.input_variable((10, 5))
    b = sample_top_k(a, k=3, num_classes=5, axis=-1)

    n = np.array([[1, 2, 3, 4, 5], ] * 10)
    n = np.stack((n,) * 1000)

    assert n.ndim == 3
    assert n.shape == (1000, 10, 5)

    results = b.eval({a: n})
    assert np.sum(results[:, :, :2]) == 0
    assert np.sum(results[:, :, 2:]) == 1000 * 10
