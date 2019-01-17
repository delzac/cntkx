import cntk as C
from cntkx.layers import QRNN, SinusoidalPositionalEmbedding, SpatialPyramidPooling
import numpy as np


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


def test_sinusoidal_positional_embedding():
    a = C.sequence.input_variable(10)
    b = SinusoidalPositionalEmbedding()(a)

    assert b.shape == (10, )

    n = np.random.random((1, 5, 10)).astype(np.float32)
    r = b.eval({a: n})

    a = C.sequence.input_variable(9)
    b = SinusoidalPositionalEmbedding()(a)

    assert b.shape == (9, )

    n = np.random.random((1, 5, 9)).astype(np.float32)
    r = b.eval({a: n})


def test_spatial_pyramid_pooling():
    # test 1
    n = np.random.random((3, 3, 32, 32)).astype(np.float32)
    a = C.input_variable((3, 32, 32))
    b = SpatialPyramidPooling((1, 2, 4))(a)

    assert b.shape == (3 * (4 * 4 + 2 * 2 + 1),)
    b.eval({a: n})

    # test 2
    n = np.random.random((3, 3, 35, 35)).astype(np.float32)
    a = C.input_variable((3, 35, 35))
    b = SpatialPyramidPooling((1, 2, 4))(a)

    assert b.shape == (3 * (4 * 4 + 2 * 2 + 1),)
    b.eval({a: n})

    # test 3
    n = np.random.random((3, 3, 35, 35)).astype(np.float32)
    a = C.input_variable((3, 35, 35))
    b = SpatialPyramidPooling((1, 3, 5))(a)

    assert b.shape == (3 * (5 * 5 + 3 * 3 + 1), )
    b.eval({a: n})

    # test 3
    n = np.random.random((3, 3, 41, 41)).astype(np.float32)
    a = C.input_variable((3, 41, 41))
    b = SpatialPyramidPooling((1, 3, 5))(a)

    assert b.shape == (3 * (5 * 5 + 3 * 3 + 1),)
    b.eval({a: n})
