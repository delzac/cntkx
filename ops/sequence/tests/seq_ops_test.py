import cntk as C
from cntkx.ops.sequence import length, pad, stride, position, join, window, reverse, reduce_mean, reduce_concat_pool
from cntkx.ops.sequence import window_causal
import numpy as np


def test_sequence_length():
    s1 = 5
    s2 = 3
    a = C.sequence.input_variable(1)
    b = length(a)
    n1 = np.random.random((s1, 1)).astype(np.float32)
    n2 = np.random.random((s2, 1)).astype(np.float32)

    result = b.eval({a: [n1, n2]})
    assert result[0][0] == s1, result[0][0]
    assert result[1][0] == s2, result[0][0]


def test_position():
    a = C.sequence.input_variable((1, 2, 3, 4, 5))
    p = position(a)

    assert p.shape == (1,)

    n = [np.random.random((4, 1, 2, 3, 4, 5)).astype(np.float32),
         np.random.random((11, 1, 2, 3, 4, 5)).astype(np.float32),
         np.random.random((15, 1, 2, 3, 4, 5)).astype(np.float32)]

    results = p.eval({a: n})

    for actual, nn in zip(results, n):
        desired = np.arange(nn.shape[0])[:, None]
        np.testing.assert_equal(actual, desired)


def test_sequence_pad():
    pattern = (1, 1)
    input_dim = 10
    s1 = 5
    s2 = 3
    a = C.sequence.input_variable(input_dim)
    b = pad(a, pattern)
    c = length(b)
    n1 = np.random.random((s1, input_dim)).astype(np.float32)
    n2 = np.random.random((s2, input_dim)).astype(np.float32)

    result = c.eval({a: [n1, n2]})
    assert result[0][0] == s1 + sum(pattern), result[0][0]
    assert result[1][0] == s2 + sum(pattern), result[0][0]
    assert b.shape == (input_dim, )


def test_sequence_pad1():
    pattern = (2, 2)
    input_dim = 10
    s1 = 5
    s2 = 3
    a = C.sequence.input_variable(input_dim)
    b = pad(a, pattern)
    c = length(b)
    n1 = np.random.random((s1, input_dim)).astype(np.float32)
    n2 = np.random.random((s2, input_dim)).astype(np.float32)

    result = c.eval({a: [n1, n2]})
    assert result[0][0] == s1 + sum(pattern), result[0][0]
    assert result[1][0] == s2 + sum(pattern), result[0][0]
    assert b.shape == (input_dim, )


def test_stride():
    contexts = [(10, 2), (10, 3), (10, 4), (10, 5), (10, 6),
                (5000, 2), (5000, 3), (5000, 4), (5000, 5), (5000, 6),
                (1000000, 2)]  # [(seq_length, stride), ...]

    a = C.sequence.input_variable(10)

    for seq_length, s in contexts:

        b = stride(a, s)

        # n = np.arange(seq_length * 1).reshape((1, seq_length, 1)).astype(np.float32)
        n = np.random.random((1, seq_length, 10)).astype(np.float32)

        output = b.eval({a: n})[0]
        np.testing.assert_equal(output, n[0][::s, ...])

    # test with mini-batch
    s = 2
    n = [np.random.random((10, 10)).astype(np.float32),
         np.random.random((5000, 10)).astype(np.float32),
         np.random.random((10000, 10)).astype(np.float32)]

    b = stride(a, s)

    output = b.eval({a: n})[-1]
    np.testing.assert_equal(output, n[-1][::s, ...])

    # test with sequence of rgb images
    a = C.sequence.input_variable((3, 64, 64))
    s = 2
    n = [np.random.random((6, 3, 64, 64)).astype(np.float32),
         np.random.random((11, 3, 64, 64)).astype(np.float32),
         np.random.random((15, 3, 64, 64)).astype(np.float32)]

    b = stride(a, s)

    output = b.eval({a: n})

    for actual, desired in zip(output, n):
        np.testing.assert_equal(actual, desired[::s])


def test_join():
    a = C.sequence.input_variable(3)
    b = C.sequence.input_variable(3)

    ab = join(a, b)

    n = [np.random.random((2, 3)).astype(np.float32),
         np.random.random((4, 3)).astype(np.float32),
         np.random.random((6, 3)).astype(np.float32), ]

    m = [np.random.random((2, 3)).astype(np.float32),
         np.random.random((4, 3)).astype(np.float32),
         np.random.random((6, 3)).astype(np.float32), ]

    joined = ab.eval({a: n, b: m})
    nm = [np.concatenate((nn, mm), axis=0) for nn, mm in zip(n, m)]

    for result, desired in zip(joined, nm):
        np.testing.assert_equal(result, desired)


def test_window():
    # ====================================================================
    # window width = 2
    # ====================================================================
    seq_length = 20
    dim = 1
    k = 2
    a = C.sequence.input_variable(dim)
    b = window(a, width=k, slide=k)

    assert b.shape == (dim * k, )

    n = np.random.random((1, seq_length, dim)).astype(np.float32)
    m = np.pad(n[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0,)

    result = b.eval({a: n})[0]
    desired = np.concatenate((n, m), axis=-1)[0, ::k]

    np.testing.assert_equal(result, desired)

    # ====================================================================
    # window width = 3
    # ====================================================================
    seq_length = 16
    dim = 4
    k = 3
    a = C.sequence.input_variable(dim)
    b = window(a, width=k, slide=k)

    assert b.shape == (dim * k,)

    n = np.random.random((1, seq_length, dim)).astype(np.float32)
    m = np.pad(n[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0, )
    o = np.pad(n[:, 2:, :], ((0, 0), (0, 2), (0, 0)), mode='constant', constant_values=0, )

    result = b.eval({a: n})[0]
    desired = np.concatenate((n, m, o), axis=-1)[0, ::k]

    np.testing.assert_equal(result, desired)


def test_window_causal():
    # ====================================================================
    # window width = 2
    # ====================================================================
    seq_length = 20
    dim = 1
    k = 2
    a = C.sequence.input_variable(dim)
    b = window_causal(a, width=k, slide=k)

    assert b.shape == (dim * k, )

    n = np.random.random((1, seq_length, dim)).astype(np.float32)
    m = np.pad(n[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0,)

    result = b.eval({a: n})[0]
    desired = np.concatenate((m, n), axis=-1)[0, ::k]

    np.testing.assert_equal(result, desired)

    # ====================================================================
    # window width = 3
    # ====================================================================
    seq_length = 16
    dim = 4
    k = 3
    a = C.sequence.input_variable(dim)
    b = window_causal(a, width=k, slide=k)

    assert b.shape == (dim * k,)

    n = np.random.random((1, seq_length, dim)).astype(np.float32)
    m = np.pad(n[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0, )
    o = np.pad(n[:, :-2, :], ((0, 0), (2, 0), (0, 0)), mode='constant', constant_values=0, )

    result = b.eval({a: n})[0]
    desired = np.concatenate((o, m, n), axis=-1)[0, ::k]

    np.testing.assert_equal(result, desired)


def test_reverse():
    ndim = 3
    a = C.sequence.input_variable(ndim)
    r = reverse(a)

    n = [np.arange(10 * ndim).reshape((10, ndim)).astype(np.float32),
         np.arange(8 * ndim).reshape((8, ndim)).astype(np.float32),
         np.arange(5 * ndim).reshape((5, ndim)).astype(np.float32), ]

    results = r.eval({a: n})

    for result, input_array in zip(results, n):
        desired = input_array[::-1, ...]
        np.testing.assert_equal(result, desired)


def test_reduce_mean():
    a = C.sequence.input_variable(32)
    b = reduce_mean(a)

    n = [np.random.random((10, 32)).astype(np.float32),
         np.random.random((10, 32)).astype(np.float32),
         np.random.random((10, 32)).astype(np.float32),]

    results = b.eval({a: n})

    for r, d in zip(results, n):
        np.testing.assert_almost_equal(r, np.mean(d, axis=0))

    a = C.sequence.input_variable((3, 4))
    b = reduce_mean(a)

    n = [np.random.random((10, 3, 4)).astype(np.float32),
         np.random.random((10, 3, 4)).astype(np.float32),
         np.random.random((10, 3, 4)).astype(np.float32),]

    results = b.eval({a: n})

    for r, d in zip(results, n):
        np.testing.assert_almost_equal(r, np.mean(d, axis=0))


def test_reduce_concat_pool():
    a = C.sequence.input_variable(32)
    b = reduce_concat_pool(a)

    assert b.shape == (32 * 3, )

    n = [np.random.random((10, 32)).astype(np.float32),
         np.random.random((10, 32)).astype(np.float32),
         np.random.random((10, 32)).astype(np.float32), ]

    b.eval({a: n})
