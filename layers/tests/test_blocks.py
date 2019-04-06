import cntk as C
import numpy as np
from cntkx.layers.blocks import WeightDroppedLSTM, IndRNN, IndyLSTM
from cntkx.layers import Recurrence


def test_weight_dropped_lstm():
    dropconnect_rate = 0.2
    variationaldrop_rate = 0.1
    a = C.sequence.input_variable(10)
    b = Recurrence(WeightDroppedLSTM(20, dropconnect_rate),
                   variational_dropout_rate_input=variationaldrop_rate,
                   variational_dropout_rate_output=variationaldrop_rate)(a)

    assert b.shape == (20, )

    n = np.random.random((2, 6, 10)).astype(np.float32)
    b.eval({a: n})


def test_ind_rnn():
    a = C.sequence.input_variable(10)
    b = Recurrence(IndRNN(20))(a)

    assert b.shape == (20,)

    n = np.random.random((2, 6, 10)).astype(np.float32)
    b.eval({a: n})


def test_ind_lstm():
    a = C.sequence.input_variable(10)
    b = Recurrence(IndyLSTM(20))(a)

    assert b.shape == (20,)

    n = np.random.random((2, 6, 10)).astype(np.float32)
    b.eval({a: n})
