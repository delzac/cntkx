import cntk as C
from cntkx.ops import cumsum, hardmax, erf, batchmatmul
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

    np.testing.assert_almost_equal(np.array(results), ans, decimal=6)

    a = C.input_variable(6)
    b = erf(a)

    n = np.array([[0, 0.02, 0.04, 0.06, 0.08, 0.1], ], dtype=np.float32)
    ans = np.array([[0, 0.022564575, 0.045111106, 0.067621594, 0.090078126, 0.112462916], ])
    results = b.eval({a: n})

    np.testing.assert_almost_equal(np.array(results), ans, decimal=6)


def test_seq_batchmatmul1():
    """ sequence axis present, left operand is matrix and right operand is matrix """
    dynamic_batch = 2
    seq_len = 4
    matmul_batch = 3

    batch_shape = (dynamic_batch, seq_len, matmul_batch)

    left_operand_shape = (5, 2)
    right_operand_shape = (2, 3)
    final_shape = (5, 3)

    n = np.random.random(batch_shape + left_operand_shape).astype(np.float32)
    m = np.random.random(batch_shape + right_operand_shape).astype(np.float32)

    a = C.sequence.input_variable((matmul_batch,) + left_operand_shape)
    b = C.sequence.input_variable((matmul_batch,) + right_operand_shape)

    c = batchmatmul(a, b)
    assert c.shape == (matmul_batch, ) + final_shape

    desired = n @ m
    result = c.eval({a: n, b: m})
    result = np.array(result)

    np.testing.assert_almost_equal(result, desired, decimal=7)


def test_seq_batchmatmul2():
    """ sequence axis present, left operand is tensor and right operand is matrix """
    dynamic_batch = 2
    seq_len = 4
    matmul_batch = 3

    batch_shape = (dynamic_batch, seq_len, matmul_batch)

    left_operand_shape = (5, 6, 2)
    right_operand_shape = (2, 3)
    final_shape = (5, 6, 3)

    n = np.random.random(batch_shape + left_operand_shape).astype(np.float32)
    m = np.random.random(batch_shape + right_operand_shape).astype(np.float32)

    a = C.sequence.input_variable((matmul_batch, ) + left_operand_shape)
    b = C.sequence.input_variable((matmul_batch, ) + right_operand_shape)

    c = batchmatmul(a, b)
    assert c.shape == (matmul_batch, ) + final_shape

    desired_packed = n.reshape(batch_shape + (-1, 2)) @ m
    desired = desired_packed.reshape(batch_shape + final_shape)
    result = c.eval({a: n, b: m})
    result = np.array(result)

    np.testing.assert_almost_equal(result, desired, decimal=7)


def test_seq_batchmatmul3():
    """ sequence axis present, left operand is matrix and right operand is tensor """
    dynamic_batch = 2
    seq_len = 4
    matmul_batch = 3

    batch_shape = (dynamic_batch, seq_len, matmul_batch)

    left_operand_shape = (5, 2)
    right_operand_shape = (2, 6, 3)
    final_shape = (5, 6, 3)

    n = np.random.random(batch_shape + left_operand_shape).astype(np.float32)
    m = np.random.random(batch_shape + right_operand_shape).astype(np.float32)

    a = C.sequence.input_variable((matmul_batch, ) + left_operand_shape)
    b = C.sequence.input_variable((matmul_batch, ) + right_operand_shape)

    c = batchmatmul(a, b, output_rank=2)
    assert c.shape == (matmul_batch, ) + final_shape

    desired_packed = n @ m.reshape(batch_shape + (2, -1))
    desired = desired_packed.reshape(batch_shape + final_shape)
    result = c.eval({a: n, b: m})
    result = np.array(result)

    np.testing.assert_almost_equal(result, desired, decimal=7)


def test_seq_batchmatmul4():
    """ No sequence axis, left operand is matrix, right operand is matrix """
    dynamic_batch = 2
    matmul_batch = 3

    batch_shape = (dynamic_batch, matmul_batch)

    inner_dim = 2
    left_operand_shape = (5, inner_dim)
    right_operand_shape = (inner_dim, 6)
    final_shape = (5, 6)

    n = np.random.random(batch_shape + left_operand_shape).astype(np.float32)
    m = np.random.random(batch_shape + right_operand_shape).astype(np.float32)

    a = C.input_variable((matmul_batch,) + left_operand_shape)
    b = C.input_variable((matmul_batch,) + right_operand_shape)

    c = batchmatmul(a, b, output_rank=1)
    assert c.shape == (matmul_batch,) + final_shape

    desired_packed = n @ m
    desired = desired_packed
    result = c.eval({a: n, b: m})
    result = np.array(result)

    np.testing.assert_almost_equal(result, desired, decimal=7)


def test_seq_batchmatmul5():
    """ No sequence axis, left operand is tensor, right operand is matrix """
    dynamic_batch = 2
    matmul_batch = 3

    batch_shape = (dynamic_batch, matmul_batch)

    inner_dim = 2
    left_operand_shape = (5, 7, inner_dim)
    right_operand_shape = (inner_dim, 6)
    final_shape = (5, 7, 6)

    n = np.random.random(batch_shape + left_operand_shape).astype(np.float32)
    m = np.random.random(batch_shape + right_operand_shape).astype(np.float32)

    a = C.input_variable((matmul_batch,) + left_operand_shape)
    b = C.input_variable((matmul_batch,) + right_operand_shape)

    c = batchmatmul(a, b, output_rank=1)
    assert c.shape == (matmul_batch,) + final_shape

    desired_packed = n.reshape(batch_shape + (-1, 2)) @ m
    desired = desired_packed.reshape(batch_shape + final_shape)
    result = c.eval({a: n, b: m})
    result = np.array(result)

    np.testing.assert_almost_equal(result, desired, decimal=7)


def test_seq_batchmatmul6():
    """ No sequence axis, left operand is matrix, right operand is tensor """
    dynamic_batch = 2
    matmul_batch = 3

    batch_shape = (dynamic_batch, matmul_batch)

    inner_dim = 2
    left_operand_shape = (5, inner_dim)
    right_operand_shape = (inner_dim, 6, 3)
    final_shape = (5, 6, 3)

    n = np.random.random(batch_shape + left_operand_shape).astype(np.float32)
    m = np.random.random(batch_shape + right_operand_shape).astype(np.float32)

    a = C.input_variable((matmul_batch,) + left_operand_shape)
    b = C.input_variable((matmul_batch,) + right_operand_shape)

    c = batchmatmul(a, b, output_rank=2)
    assert c.shape == (matmul_batch, ) + final_shape

    desired_packed = n @ m.reshape(batch_shape + (inner_dim, -1))
    desired = desired_packed.reshape(batch_shape + final_shape)
    result = c.eval({a: n, b: m})
    result = np.array(result)

    np.testing.assert_almost_equal(result, desired, decimal=7)


def test_seq_batchmatmul7():
    """
    sequence axis present with samples of uneven sequence length
    left operand is matrix, right operand is matrix
    """

    # uneven sequence length between batches
    n = [np.random.random((4,   3,   5, 2)).astype(np.float32),
         np.random.random((2,   3,   5, 2)).astype(np.float32),
         np.random.random((7,   3,   5, 2)).astype(np.float32)]

    m = [np.random.random((4,   3,   2, 7)).astype(np.float32),
         np.random.random((2,   3,   2, 7)).astype(np.float32),
         np.random.random((7,   3,   2, 7)).astype(np.float32)]

    a = C.sequence.input_variable((3, 5, 2))
    b = C.sequence.input_variable((3, 2, 7))

    c = batchmatmul(a, b)
    assert c.shape == (3, 5, 7)

    desired_packed = [nn @ mm for nn, mm in zip(n, m)]
    desired_results = desired_packed
    results = c.eval({a: n, b: m})

    assert len(results) == len(desired_results)

    for result, desired in zip(results, desired_results):
        np.testing.assert_almost_equal(result, desired, decimal=7)
