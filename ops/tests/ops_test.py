import cntk as C
from cntkx.ops import cumsum, scaled_dot_product_attention
import numpy as np
from numpy.testing import assert_equal, assert_raises


def test_cumsum():
    a = C.input_variable(5)
    b = cumsum(a)

    n = np.array([1, 2, 3, 4, 5]).astype(np.float32)[None, ...]
    results = b.eval({a: n})
    assert_equal(results[0], n.cumsum())


def test_scaled_dot_product_attention1():
    """ check that function can output as non-sequence and sequence """
    s1, s2 = 4, 2
    a = C.sequence.input_variable(5)
    b = scaled_dot_product_attention(a, a, a)

    n1 = np.random.random((s1, 5)).astype(np.float32)
    n2 = np.random.random((s2, 5)).astype(np.float32)

    results = b.eval({a: [n1, n2]})
    assert_equal(np.sum(results[1][s2:]), 0)  # check that n2 after seq_len 2 is empty
    assert results[1].shape != n2.shape and results[1].shape == n1.shape, f"Wrong expected shape {results[1].shape} != {n1.shape}"

    b = scaled_dot_product_attention(a, a, a, output_as_seq=True)
    results = b.eval({a: [n1, n2]})
    assert results[1].shape == n2.shape, f"Wrong expected shape {results[1].shape} != {n2.shape}"


def test_scaled_dot_product_attention2():
    """ check that valid mask passed in as argument into function works """
    a = C.input_variable((6, 5))
    valid = C.input_variable(6)
    b = scaled_dot_product_attention(a, a, a, valid_mask_value=valid)

    n = np.random.random((2, 6, 5)).astype(np.float32)
    v = np.array([[1, 1, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0]]).astype(np.float32)

    results = b.eval({a: n, valid: v})
    assert_equal(np.sum(results[0][2:]), 0)
    assert_raises(AssertionError, assert_equal, np.sum(results[0][1:]), 0)

    assert_equal(np.sum(results[1][4:]), 0)
    assert_raises(AssertionError, assert_equal, np.sum(results[1][3:]), 0)


def test_scaled_dot_product_attention3():
    """ check that function can work while not peeking on future values """
    s1, s2 = 4, 2
    a = C.sequence.input_variable(5)
    b = scaled_dot_product_attention(a, a, a, obey_sequence_order=True, max_seq_len=100, output_as_seq=True)

    n1 = np.random.random((s1, 5)).astype(np.float32)
    n2 = np.random.random((s2, 5)).astype(np.float32)

    results = b.eval({a: [n1, n2]})
    assert results[1].shape == n2.shape, f"Wrong expected shape {results[1].shape} != {n2.shape}"


def test_scaled_dot_product_attention4():
    """ check that return valid_mask of value works """
    s1, s2 = 4, 2
    a = C.sequence.input_variable(5)
    b, v = scaled_dot_product_attention(a, a, a, obey_sequence_order=True, max_seq_len=100, return_valid_mask=True)

    n1 = np.random.random((s1, 5)).astype(np.float32)
    n2 = np.random.random((s2, 5)).astype(np.float32)

    results = b.eval({a: [n1, n2]})
    assert_equal(np.sum(results[1][s2:]), 0)  # check that n2 after seq_len 2 is empty
    assert results[1].shape == n1.shape, f"Wrong expected shape {results[1].shape} != {n1.shape}"

    v = v + 0
    results = v.eval({a: [n1, n2]})
    assert_equal(np.sum(results[1]), s2)
    assert_equal(np.sum(results[0]), s1)
