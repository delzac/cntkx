import cntk as C
from cntkx.ops.sequence import length, pad
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
