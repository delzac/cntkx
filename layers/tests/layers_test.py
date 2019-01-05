import cntk as C
from cntkx.layers import QRNN, MultiheadAttention
import numpy as np
import pytest


def test_qrnn():
    HIDDEN_DIM = 20
    i = C.sequence.input_variable(10)
    qrnn = QRNN(hidden_dim=HIDDEN_DIM)(i)

    assert qrnn.shape[0] == HIDDEN_DIM


def test_qrnn1():
    input_dim = 3
    even_length = 10
    odd_length = 15
    i = C.sequence.input_variable(input_dim)
    qrnn = QRNN(hidden_dim=50)(i)

    n1 = np.random.random((odd_length, input_dim)).astype(np.float32)
    n2 = np.random.random((even_length, input_dim)).astype(np.float32)

    qrnn.eval({i: [n1, n2]})


def test_multi_head_attention1():
    """ default settings: output as unpacked sequence tensor """
    a = C.sequence.input_variable(5)
    multihead1 = MultiheadAttention(nb_heads=6, model_dim=30)
    multihead2 = MultiheadAttention(nb_heads=6, model_dim=60, map_rank=1)

    attended = multihead1(a, a, a)
    assert attended.shape == (-3, 30)

    attended = multihead2(attended, attended, attended, a)
    assert attended.shape == (-3, 60)

    n = [np.random.random((2, 5)).astype(np.float32),
         np.random.random((4, 5)).astype(np.float32),
         np.random.random((6, 5)).astype(np.float32)]

    results = attended.eval({a: n})
    np.testing.assert_equal(results[0][2:], 0)
    np.testing.assert_equal(results[1][4:], 0)

    with pytest.raises(Exception):
        np.testing.assert_equal(results[1][2:], 0)
        np.testing.assert_equal(results[2][4:], 0)


def test_multi_head_attention1a():
    """ default settings: output as unpacked sequence tensor """
    a = C.sequence.input_variable(5)
    multihead1 = MultiheadAttention(nb_heads=6, model_dim=30)
    multihead2 = MultiheadAttention(nb_heads=6, model_dim=60, map_rank=1)

    attended = multihead1(a, a, a)
    assert attended.shape == (-3, 30)

    attended = multihead2(attended, attended, attended, None)
    assert attended.shape == (-3, 60)

    n = [np.random.random((2, 5)).astype(np.float32),
         np.random.random((4, 5)).astype(np.float32),
         np.random.random((6, 5)).astype(np.float32)]

    results = attended.eval({a: n})
    with pytest.raises(Exception):
        np.testing.assert_equal(results[0][2:], 0)
        np.testing.assert_equal(results[1][4:], 0)


def test_multi_head_attention2():
    """ default settings: output as sequence tensor """
    a = C.sequence.input_variable(5)
    multihead1 = MultiheadAttention(nb_heads=6, model_dim=30, output_as_seq=True)
    multihead2 = MultiheadAttention(nb_heads=6, model_dim=60, output_as_seq=True)

    attended = multihead1(a, a, a)
    assert attended.shape == (30,)

    attended = multihead2(attended, attended, attended)
    assert attended.shape == (60,)
    assert attended.dynamic_axes == a.dynamic_axes

    n = np.random.random((2, 3, 5)).astype(np.float32)
    attended.eval({a: n})


def test_multi_head_attention3():
    """ default settings: output as sequence tensor with no peeking into future values """
    a = C.sequence.input_variable(5)
    multihead1 = MultiheadAttention(nb_heads=6, model_dim=30, obey_sequence_order=True,
                                    max_seq_len=10, output_as_seq=True)

    attended = multihead1(a, a, a)
    assert attended.shape == (30,)

    n = np.random.random((2, 3, 5)).astype(np.float32)
    attended.eval({a: n})

    n = np.random.random((2, 11, 5)).astype(np.float32)
    with pytest.raises(Exception):
        attended.eval({a: n})
