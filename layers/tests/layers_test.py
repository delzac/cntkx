import cntk as C
from cntkx.layers import QRNN, SinusoidalPositionalEmbedding, SpatialPyramidPooling, GatedLinearUnit
from cntkx.layers import WeightDroppedLSTM, BertEmbeddings, PositionalEmbedding
from cntkx.layers import PreTrainedBertEmbeddings, PositionwiseFeedForward, SequentialMaxPooling
from cntkx.layers import SequentialStride
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


def test_gated_linear_unit():
    input_dim = 10
    even_length = 10
    odd_length = 15

    a = C.sequence.input_variable(input_dim)
    b = GatedLinearUnit(2, 20)(a)

    assert b.shape == (20, )

    n1 = np.random.random((odd_length, input_dim)).astype(np.float32)
    n2 = np.random.random((even_length, input_dim)).astype(np.float32)

    b.eval({a: [n1, n2]})


def test_weight_dropped_lstm():
    a = C.sequence.input_variable(10)
    b = WeightDroppedLSTM(20, 0.1, 0.1, 0.1)(a)

    assert b.shape == (20, )

    n1 = np.random.random((3, 10)).astype(np.float32)
    n2 = np.random.random((6, 10)).astype(np.float32)

    b.eval({a: [n1, n2]})

    b = WeightDroppedLSTM(20, 0, 0, 0)(a)

    assert b.shape == (20,)

    n1 = np.random.random((3, 10)).astype(np.float32)
    n2 = np.random.random((6, 10)).astype(np.float32)

    b.eval({a: [n1, n2]})


def test_positional_embedding():
    max_seq_length = 100
    hidden_dim = 120
    a = C.sequence.input_variable(12)
    b = PositionalEmbedding(max_seq_length, hidden_dim)(a)

    assert b.shape == (hidden_dim, )

    n1 = np.random.random((3, 12)).astype(np.float32)
    n2 = np.random.random((6, 12)).astype(np.float32)
    b.eval({a: [n1, n2]})


def test_bert_embeddings():
    max_seq_length = 512
    hidden_dim = 768

    text_tensor = C.sequence.input_variable(100)
    token_type_tensor = C.sequence.input_variable(2)
    b = BertEmbeddings(max_seq_length, hidden_dim, 0.1)(text_tensor, token_type_tensor)

    assert b.shape == (hidden_dim, )

    n1 = np.random.random((3, 100)).astype(np.float32)
    n2 = np.random.random((6, 100)).astype(np.float32)

    m1 = np.random.random((3, 2)).astype(np.float32)
    m2 = np.random.random((6, 2)).astype(np.float32)
    b.eval({text_tensor: [n1, n2], token_type_tensor: [m1, m2]})


def test_pretrained_bert_embeddings_learnable():
    """ tested to work with 'uncased_L-12_H-768_A-12' """
    text_tensor = C.sequence.input_variable(30522)
    token_type_tensor = C.sequence.input_variable(2)
    filepath_to_tf_bert_model = "../../../pretrained models/BERT/uncased/bert_model.ckpt"

    embeddings = PreTrainedBertEmbeddings(filepath_to_tf_bert_model, 0.1, name='bert')
    b = embeddings(text_tensor, token_type_tensor)

    assert b.shape == (768,)

    n1 = np.random.random((3, 30522)).astype(np.float32)
    n2 = np.random.random((6, 30522)).astype(np.float32)

    m1 = np.random.random((3, 2)).astype(np.float32)
    m2 = np.random.random((6, 2)).astype(np.float32)
    b.eval({text_tensor: [n1, n2], token_type_tensor: [m1, m2]})

    embed = b.bert.word_embeddings
    assert embed.shape == (768,)

    embed = b.bert.position_embeddings
    assert embed.parameters[0].shape == (512, 768)

    embed = b.bert.token_type_embeddings
    assert embed.parameters[0].shape == (2, 768)


def test_positionwise_feedforward():
    model_dim = 20
    inner_dim = 30

    a = C.sequence.input_variable(10)
    b = PositionwiseFeedForward(model_dim, inner_dim, 0.1)(a)

    assert b.shape == (model_dim, )

    n1 = np.random.random((3, 10)).astype(np.float32)
    n2 = np.random.random((6, 10)).astype(np.float32)

    b.eval({a: [n1, n2]})


def test_sequential_stride():
    # stride = 1, confirm no change in between input and output
    a = C.sequence.input_variable((3, 10))
    stride = SequentialStride(input_ndim=2, dim_axis0=3, stride=1, pad=False)
    b = stride(a)

    n = np.ascontiguousarray(np.arange(3 * 6 * 1 * 10).reshape((1, 6, 3, 10)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    np.testing.assert_almost_equal(output, np.squeeze(n))

    # stride = 2
    a = C.sequence.input_variable((3, 10))
    stride = SequentialStride(input_ndim=2, dim_axis0=3, stride=2, pad=False)
    b = stride(a)

    n = np.ascontiguousarray(np.arange(3 * 6 * 1 * 10).reshape((1, 6, 3, 10)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    np.testing.assert_almost_equal(output, np.squeeze(n)[::2])

    # stride = 3 with seq of 3d tensor
    a = C.sequence.input_variable((3, 10, 15))
    stride = SequentialStride(input_ndim=3, dim_axis0=3, stride=3, pad=False)
    b = stride(a)

    n = np.ascontiguousarray(np.arange(3 * 6 * 15 * 10).reshape((1, 6, 3, 10, 15)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    np.testing.assert_almost_equal(output, np.squeeze(n)[::3])

    # stride = 3 with seq of 1d vector
    a = C.sequence.input_variable((3,))
    stride = SequentialStride(input_ndim=1, dim_axis0=3, stride=3, pad=False)
    b = stride(a)

    n = np.ascontiguousarray(np.arange(3 * 6 * 1 * 1).reshape((1, 6, 3)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    np.testing.assert_almost_equal(output, np.squeeze(n)[::3])


def test_sequential_max_pooling1():
    a = C.sequence.input_variable((2, 4))
    b = SequentialMaxPooling(filter_shape=(2, 2), strides=(2, 2), pad=False)(a)

    n = np.ascontiguousarray(np.arange(2 * 5 * 4).reshape((1, 5, 2, 4)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    a = C.input_variable((2, 4, 5))
    b = C.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2), pad=False)(a)

    n = np.arange(2 * 5 * 4).reshape((1, 5, 2, 4)).astype(np.float32)
    n = np.ascontiguousarray(np.moveaxis(n, 1, -1))

    desired = b.eval({a: n})
    desired = np.squeeze(np.moveaxis(desired, -1, 1))

    np.testing.assert_almost_equal(output[:2, ...], desired)

    # BUGBUG: Sequential maxpooling will 'right pad' on sequential axis
    # BUGBUG: once fixed, this assertion should fail
    assert output.shape[0] == 3 and desired.shape[0] == 2
    assert output.shape[0] != desired.shape[0], "Due to bug, sequence length is different between desired and output"


def test_sequential_max_pooling2():
    a = C.sequence.input_variable((3, 25))
    b = SequentialMaxPooling(filter_shape=(2, 2), strides=(2, 2), pad=False)(a)

    assert b.shape == (3, 12)

    n = np.ascontiguousarray(np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    a = C.input_variable((3, 25, 6))
    b = C.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2), pad=False)(a)

    n = np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32)
    n = np.ascontiguousarray(np.moveaxis(n, 1, -1))

    desired = b.eval({a: n})
    desired = np.squeeze(np.moveaxis(desired, -1, 1))

    np.testing.assert_almost_equal(output, desired)


def test_sequential_max_pooling3():
    a = C.sequence.input_variable((3, 25))
    b = SequentialMaxPooling(filter_shape=(3, 3), strides=(2, 2), pad=False)(a)

    n = np.ascontiguousarray(np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    a = C.input_variable((3, 25, 6))
    b = C.layers.MaxPooling(filter_shape=(3, 3), strides=(2, 2), pad=False)(a)

    n = np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32)
    n = np.ascontiguousarray(np.moveaxis(n, 1, -1))

    desired = b.eval({a: n})
    desired = np.squeeze(np.moveaxis(desired, -1, 1))

    # BUGBUG: Sequential maxpooling will 'right pad' on sequential axis
    # BUGBUG: once fixed, this assertion should fail
    np.testing.assert_almost_equal(output[:2], desired)


def test_sequential_max_pooling4():
    a = C.sequence.input_variable((3, 25))
    b = SequentialMaxPooling(filter_shape=(3, 3), strides=(2, 2), pad=True)(a)

    assert b.shape == (3, 13)

    n = np.ascontiguousarray(np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    a = C.input_variable((3, 25, 6))
    b = C.layers.MaxPooling(filter_shape=(3, 3), strides=(2, 2), pad=True)(a)

    n = np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32)
    n = np.ascontiguousarray(np.moveaxis(n, 1, -1))

    desired = b.eval({a: n})
    desired = np.squeeze(np.moveaxis(desired, -1, 1))

    np.testing.assert_almost_equal(output, desired)


def test_sequential_max_pooling5():
    a = C.sequence.input_variable((3, 25))
    b = SequentialMaxPooling(filter_shape=(4, 4), strides=(2, 2), pad=True)(a)

    n = np.ascontiguousarray(np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    a = C.input_variable((3, 25, 6))
    b = C.layers.MaxPooling(filter_shape=(4, 4), strides=(2, 2), pad=True)(a)

    n = np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32)
    n = np.ascontiguousarray(np.moveaxis(n, 1, -1))

    desired = b.eval({a: n})
    desired = np.squeeze(np.moveaxis(desired, -1, 1))

    np.testing.assert_almost_equal(output, desired)


def test_sequential_max_pooling6():
    """ sequential max pool across a sequence of vector """
    a = C.sequence.input_variable((25, ))
    b = SequentialMaxPooling(filter_shape=(4,), strides=(2,), pad=True)(a)

    n = np.ascontiguousarray(np.arange(1 * 6 * 25).reshape((1, 6, 25)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    a = C.input_variable((25, 6))
    b = C.layers.MaxPooling(filter_shape=(4, ), strides=(2, ), pad=True)(a)

    n = np.arange(1 * 6 * 25).reshape((1, 6, 25)).astype(np.float32)
    n = np.ascontiguousarray(np.moveaxis(n, 1, -1))

    desired = b.eval({a: n})
    desired = np.squeeze(np.moveaxis(desired, -1, 1))

    np.testing.assert_almost_equal(output, desired)
