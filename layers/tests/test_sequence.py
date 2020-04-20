import cntk as C
import numpy as np
from cntkx.layers.sequence import Recurrence, VariationalDropout, PyramidalBiRecurrence, BiRecurrence, SequenceDropout
from cntkx.layers import IndyLSTM
from cntk.layers import LSTM


def test_recurrence():
    # No variational dropout
    a = C.sequence.input_variable(10)
    b = Recurrence(C.plus)(a)

    n = np.arange(100).reshape((1, 10, 10)).astype(np.float32)
    result = b.eval({a: n})
    result = np.array(result)
    desired = np.cumsum(n, 1, dtype=np.float32)

    np.testing.assert_equal(result, desired)

    # -------------------------------------------------------------------------
    # input variational dropout - total dropout
    # -------------------------------------------------------------------------
    a = C.sequence.input_variable(10)
    b = Recurrence(C.plus, dropout_rate_input=.9999999999)(a)

    n = np.arange(100).reshape((1, 10, 10)).astype(np.float32)
    __, result = b.forward({a: n}, [b.output], set([b.output]))
    result = np.array(list(result.values())[0])
    desired = np.cumsum(n, 1, dtype=np.float32) * 0

    np.testing.assert_equal(result, desired)

    # -------------------------------------------------------------------------
    # output variational dropout - total dropout
    # -------------------------------------------------------------------------
    a = C.sequence.input_variable(10)
    b = Recurrence(C.plus, dropout_rate_output=.9999999999)(a)

    n = np.arange(100).reshape((1, 10, 10)).astype(np.float32)
    __, result = b.forward({a: n}, [b.output], set([b.output]))
    result = np.array(list(result.values())[0])
    desired = np.cumsum(n, 1, dtype=np.float32) * 0

    np.testing.assert_equal(result, desired)

    # -------------------------------------------------------------------------
    # input variational dropout - Zero dropout
    # -------------------------------------------------------------------------
    a = C.sequence.input_variable(10)
    b = Recurrence(C.plus, dropout_rate_input=.000000001)(a)

    n = np.arange(100).reshape((1, 10, 10)).astype(np.float32)
    __, result = b.forward({a: n}, [b.output], set([b.output]))
    result = np.array(list(result.values())[0])
    desired = np.cumsum(n, 1, dtype=np.float32)

    np.testing.assert_equal(result, desired)

    # -------------------------------------------------------------------------
    # output variational dropout - Zero dropout
    # -------------------------------------------------------------------------
    a = C.sequence.input_variable(10)
    b = Recurrence(C.plus, dropout_rate_output=.000000001)(a)

    n = np.arange(100).reshape((1, 10, 10)).astype(np.float32)
    __, result = b.forward({a: n}, [b.output], set([b.output]))
    result = np.array(list(result.values())[0])
    desired = np.cumsum(n, 1, dtype=np.float32)

    np.testing.assert_equal(result, desired)

    # -------------------------------------------------------------------------
    # output variational dropout - half dropout
    # -------------------------------------------------------------------------
    a = C.sequence.input_variable(10)
    b = Recurrence(C.plus, dropout_rate_output=.5, seed=12)(a)

    n = np.arange(100).reshape((1, 10, 10)).astype(np.float32)
    __, result = b.forward({a: n}, [b.output], set([b.output]))
    result = np.array(list(result.values())[0])
    desired = np.cumsum(n, 1, dtype=np.float32) * 1 / 0.5  # cntk scales the input by 1/dropout

    matched = np.sum(np.equal(result, desired))
    assert matched == 50 + 1  # + 1 for the first element of the first sequence


def test_pyramidal_bi_recurrence():
    dim = 10
    width = 2
    hidden_dim = 30
    seq_length = 16
    a = C.sequence.input_variable(dim)
    b = PyramidalBiRecurrence(LSTM(hidden_dim), LSTM(hidden_dim), width)(a)

    assert b.shape == (hidden_dim * 2 * width, )

    n = np.random.random((1, 16, 10)).astype(np.float32)
    result = b.eval({a: n})[0]

    assert result.shape == (seq_length / width, hidden_dim * 2 * width)


def test_variational_dropout():
    dim = 10
    dropout_rate = 0.5
    a = C.sequence.input_variable(dim)
    b = VariationalDropout(dropout_rate, seed=12)(a)

    n = np.arange(100).reshape((1, 10, 10)).astype(np.float32)
    __, result = b.forward({a: n}, [b.output], set([b.output]))  # simulate training, dropout during inference is no-op

    result = list(result.values())[0][0]

    np.testing.assert_equal(np.sum(np.equal(result.sum(axis=0), 0)), dim * dropout_rate)


def test_birecurrence():
    dim = 10
    hidden_dim = 30

    a = C.sequence.input_variable(dim)
    b = BiRecurrence(LSTM(hidden_dim), weight_tie=False)(a)

    assert b.shape == (hidden_dim * 2,)

    c = BiRecurrence(LSTM(hidden_dim), weight_tie=True)(a)

    assert c.shape == b.shape
    assert len(b.parameters) == 3 + 3
    assert len(c.parameters) == 3 + 4
    assert c.f_token0.shape == c.b_token0.shape == (hidden_dim, )
    assert c.f_token1.shape == c.b_token1.shape == (hidden_dim, )

    d = BiRecurrence(IndyLSTM(hidden_dim), weight_tie=True)(a)

    assert d.shape == b.shape
    assert len(b.parameters) == 3 + 3
    assert len(d.parameters) == 3 + 4
    assert d.f_token0.shape == d.b_token0.shape == (hidden_dim, )
    assert d.f_token1.shape == d.b_token1.shape == (hidden_dim, )

    n = [np.random.random((5, 10)).astype(np.float32),
         np.random.random((7, 10)).astype(np.float32),]

    c.eval({a: n})
    b.eval({a: n})


def test_sequence_dropout():
    desired = [5, 6, 19, 24, 53, 259]
    dim = 3
    a = C.sequence.input_variable(dim)
    b = SequenceDropout(0.5, 1)(a)

    n = [np.ones((10, dim)).astype(np.float32),
         np.ones((20, dim)).astype(np.float32),
         np.ones((30, dim)).astype(np.float32),
         np.ones((50, dim)).astype(np.float32),
         np.ones((100, dim)).astype(np.float32),
         np.ones((500, dim)).astype(np.float32), ]

    df, fv = b.forward({a: n}, [b.output], set([b.output]))

    for seq, d in zip(fv[b.output], desired):
        non_zeroed = np.count_nonzero(np.mean(seq, axis=1))
        assert non_zeroed == d
