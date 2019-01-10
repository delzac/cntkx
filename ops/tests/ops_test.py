import cntk as C
from cntkx.ops import cumsum, scaled_dot_product_attention, hardmax
import numpy as np
from numpy.testing import assert_equal, assert_raises
import pytest


def test_cumsum():
    a = C.input_variable(5)
    b = cumsum(a)

    n = np.array([1, 2, 3, 4, 5]).astype(np.float32)[None, ...]
    results = b.eval({a: n})
    assert_equal(results[0], n.cumsum())


def test_scaled_dot_product_attention1a():
    """ check default works. returns non-sequence with input being a sequence """
    s1, s2 = 4, 2
    a = C.sequence.input_variable(5)
    b = scaled_dot_product_attention(a, a, a)

    n1 = np.random.random((s1, 5)).astype(np.float32)
    n2 = np.random.random((s2, 5)).astype(np.float32)

    results = b.eval({a: [n1, n2]})
    assert_equal(np.sum(results[1][s2:]), 0)  # check that n2 after seq_len 2 is empty
    assert results[1].shape != n2.shape and results[1].shape == n1.shape, f"Wrong expected shape {results[1].shape} != {n1.shape}"


def test_scaled_dot_product_attention1b():
    """ returns sequence with input being a sequence """
    s1, s2 = 4, 2
    a = C.sequence.input_variable(5)
    b = scaled_dot_product_attention(a, a, a, output_as_seq=True)

    n1 = np.random.random((s1, 5)).astype(np.float32)
    n2 = np.random.random((s2, 5)).astype(np.float32)

    results = b.eval({a: [n1, n2]})
    assert results[1].shape == n2.shape, f"Wrong expected shape {results[1].shape} != {n2.shape}"


def test_scaled_dot_product_attention2():
    """ output as non-sequence when input is not a sequence but dynamic axis is provided """
    a = C.input_variable((6, 5))
    seq = C.sequence.input_variable(5)
    b = scaled_dot_product_attention(a, a, a, dynamic_axes_like=seq)

    n = np.random.random((3, 6, 5)).astype(np.float32)
    v = [np.random.random((2, 5)).astype(np.float32),
         np.random.random((4, 5)).astype(np.float32),
         np.random.random((6, 5)).astype(np.float32)]

    results = b.eval({a: n, seq: v})
    assert_equal(np.sum(results[0][2:]), 0)
    assert_raises(AssertionError, assert_equal, np.sum(results[0][1:]), 0)

    assert_equal(np.sum(results[1][4:]), 0)
    assert_raises(AssertionError, assert_equal, np.sum(results[1][3:]), 0)


def test_scaled_dot_product_attention3():
    """ returns a sequence while not peeking on future values """
    s1, s2 = 4, 2
    a = C.sequence.input_variable(5)
    b = scaled_dot_product_attention(a, a, a, obey_sequence_order=True, max_seq_len=100, output_as_seq=True)

    n1 = np.random.random((s1, 5)).astype(np.float32)
    n2 = np.random.random((s2, 5)).astype(np.float32)

    results = b.eval({a: [n1, n2]})
    assert results[1].shape == n2.shape, f"Wrong expected shape {results[1].shape} != {n2.shape}"


def test_scaled_dot_product_attention4():
    """ returns a non-sequence while not peeking on future values """
    s1, s2 = 4, 2
    a = C.sequence.input_variable(5)
    b = scaled_dot_product_attention(a, a, a, obey_sequence_order=True, max_seq_len=100)

    n1 = np.random.random((s1, 5)).astype(np.float32)
    n2 = np.random.random((s2, 5)).astype(np.float32)

    results = b.eval({a: [n1, n2]})
    assert_equal(np.sum(results[1][s2:]), 0)  # check that n2 after seq_len 2 is empty
    assert results[1].shape == n1.shape, f"Wrong expected shape {results[1].shape} != {n1.shape}"


def test_scaled_dot_product_attention5():
    """ if input is sequence, dynamic_axes_like don't need to be seq """
    a = C.sequence.input_variable(5)
    with pytest.raises(Exception):
        scaled_dot_product_attention(a, a, a, dynamic_axes_like=a)


def test_scaled_dot_product_attention6():
    """ query and key is non-seq and value is seq """
    a = C.sequence.input_variable(5)
    b = C.input_variable((-3, 5))

    c = scaled_dot_product_attention(b, b, a)

    n = np.random.random((3, 6, 5)).astype(np.float32)
    m = [np.random.random((2, 5)).astype(np.float32),
         np.random.random((4, 5)).astype(np.float32),
         np.random.random((6, 5)).astype(np.float32)]

    results = c.eval({a: m, b: n})


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
