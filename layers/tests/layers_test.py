import cntk as C
import cntkx as Cx
from cntkx.layers import QRNN, SinusoidalPositionalEmbedding, SpatialPyramidPooling, GatedLinearUnit
from cntkx.layers import BertEmbeddings, PositionalEmbedding, SequentialAveragePooling
from cntkx.layers import PreTrainedBertEmbeddings, PositionwiseFeedForward, SequentialMaxPooling
from cntkx.layers import vFSMN, cFSMN, SequentialConcatPooling, SequentialDense
import numpy as np
import math


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
    seq = 50
    dim = 100
    a = C.sequence.input_variable(dim)
    b = SinusoidalPositionalEmbedding(dim)(a)

    assert b.shape == (dim, )

    n = np.random.random((1, seq, dim)).astype(np.float32)
    r = b.eval({a: n})

    dim = 99
    a = C.sequence.input_variable(dim)
    b = SinusoidalPositionalEmbedding(dim)(a)

    assert b.shape == (dim, )

    n = np.random.random((1, seq, dim)).astype(np.float32)
    r = b.eval({a: n})

    assert np.sum(r[0][:, -1]) == 0, "last dimension should be all zeros if dimension is odd"

    # plt.imshow(r[0])  # to view image to confirm correctness
    # plt.show()


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


def test_positional_embedding():
    max_seq_length = 100
    hidden_dim = 120
    a = C.sequence.input_variable(12)
    b = PositionalEmbedding(hidden_dim, max_seq_length)(a)

    assert b.shape == (hidden_dim, )

    n1 = np.random.random((3, 12)).astype(np.float32)
    n2 = np.random.random((6, 12)).astype(np.float32)
    b.eval({a: [n1, n2]})


def test_bert_embeddings():
    max_seq_length = 512
    hidden_dim = 768

    text_tensor = C.sequence.input_variable(100)
    token_type_tensor = C.sequence.input_variable(2)
    b = BertEmbeddings(hidden_dim, max_seq_length, 0.1)(text_tensor, token_type_tensor)

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


def test_sequential_max_pooling():
    # ===================================================================
    # sequential max pool across rgb images with width in sequence axis
    # ===================================================================
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

    # ===================================================================
    # sequential max pool across rgb images with width in sequence axis
    # ===================================================================
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

    # ===================================================================
    # sequential max pool across rgb images with width in sequence axis
    # ===================================================================
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

    # ===================================================================
    # sequential max pool across rgb images with width in sequence axis
    # ===================================================================
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

    # ===================================================================
    # sequential max pool across rgb images with width in sequence axis
    # ===================================================================
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

    # ===================================================================
    # sequential max pool across a sequence of vector or B&W images
    # ===================================================================
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


def test_sequential_average_pooling():
    # ===================================================================
    # sequential max pool across rgb images with width in sequence axis
    # ===================================================================
    a = C.sequence.input_variable((2, 4))
    b = SequentialAveragePooling(filter_shape=(2, 2), strides=(2, 2), pad=False)(a)

    n = np.ascontiguousarray(np.arange(2 * 5 * 4).reshape((1, 5, 2, 4)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    a = C.input_variable((2, 4, 5))
    b = C.layers.AveragePooling(filter_shape=(2, 2), strides=(2, 2), pad=False)(a)

    n = np.arange(2 * 5 * 4).reshape((1, 5, 2, 4)).astype(np.float32)
    n = np.ascontiguousarray(np.moveaxis(n, 1, -1))

    desired = b.eval({a: n})
    desired = np.squeeze(np.moveaxis(desired, -1, 1))

    np.testing.assert_almost_equal(output[:2, ...], desired)

    # BUGBUG: Sequential AveragePooling will 'right pad' on sequential axis
    # BUGBUG: once fixed, this assertion should fail
    assert output.shape[0] == 3 and desired.shape[0] == 2
    assert output.shape[0] != desired.shape[0], "Due to bug, sequence length is different between desired and output"

    # ===================================================================
    # sequential max pool across rgb images with width in sequence axis
    # ===================================================================
    a = C.sequence.input_variable((3, 25))
    b = SequentialAveragePooling(filter_shape=(2, 2), strides=(2, 2), pad=False)(a)

    assert b.shape == (3, 12)

    n = np.ascontiguousarray(np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    a = C.input_variable((3, 25, 6))
    b = C.layers.AveragePooling(filter_shape=(2, 2), strides=(2, 2), pad=False)(a)

    n = np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32)
    n = np.ascontiguousarray(np.moveaxis(n, 1, -1))

    desired = b.eval({a: n})
    desired = np.squeeze(np.moveaxis(desired, -1, 1))

    np.testing.assert_almost_equal(output, desired)

    # ===================================================================
    # sequential max pool across rgb images with width in sequence axis
    # ===================================================================
    a = C.sequence.input_variable((3, 25))
    b = SequentialAveragePooling(filter_shape=(3, 3), strides=(2, 2), pad=False)(a)

    n = np.ascontiguousarray(np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    a = C.input_variable((3, 25, 6))
    b = C.layers.AveragePooling(filter_shape=(3, 3), strides=(2, 2), pad=False)(a)

    n = np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32)
    n = np.ascontiguousarray(np.moveaxis(n, 1, -1))

    desired = b.eval({a: n})
    desired = np.squeeze(np.moveaxis(desired, -1, 1))

    # BUGBUG: Sequential AveragePooling will 'right pad' on sequential axis
    # BUGBUG: once fixed, this assertion should fail
    np.testing.assert_almost_equal(output[:2], desired)

    # ===================================================================
    # sequential max pool across rgb images with width in sequence axis
    # ===================================================================
    a = C.sequence.input_variable((3, 25))
    b = SequentialAveragePooling(filter_shape=(3, 3), strides=(2, 2), pad=True)(a)

    assert b.shape == (3, 13)

    n = np.ascontiguousarray(np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    a = C.input_variable((3, 25, 6))
    b = C.layers.AveragePooling(filter_shape=(3, 3), strides=(2, 2), pad=True)(a)

    n = np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32)
    n = np.ascontiguousarray(np.moveaxis(n, 1, -1))

    desired = b.eval({a: n})
    desired = np.squeeze(np.moveaxis(desired, -1, 1))

    # ignore the first seq item because SequentialAveragePooling included padding in average calculation
    np.testing.assert_almost_equal(output[1:], desired[1:])

    # ===================================================================
    # sequential max pool across rgb images with width in sequence axis
    # ===================================================================
    a = C.sequence.input_variable((3, 25))
    b = SequentialAveragePooling(filter_shape=(4, 4), strides=(2, 2), pad=True)(a)

    n = np.ascontiguousarray(np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    a = C.input_variable((3, 25, 6))
    b = C.layers.AveragePooling(filter_shape=(4, 4), strides=(2, 2), pad=True)(a)

    n = np.arange(3 * 6 * 25).reshape((1, 6, 3, 25)).astype(np.float32)
    n = np.ascontiguousarray(np.moveaxis(n, 1, -1))

    desired = b.eval({a: n})
    desired = np.squeeze(np.moveaxis(desired, -1, 1))

    # ignore the first and last seq item because SequentialAveragePooling included padding in average calculation
    np.testing.assert_almost_equal(output[1:-1], desired[1:-1])

    # ===================================================================
    # sequential max pool across a sequence of vector or B&W images
    # ===================================================================
    a = C.sequence.input_variable((25, ))
    b = SequentialAveragePooling(filter_shape=(4,), strides=(2,), pad=True)(a)

    n = np.ascontiguousarray(np.arange(1 * 6 * 25).reshape((1, 6, 25)).astype(np.float32))
    output = b.eval({a: n})

    assert isinstance(output, list) and len(output) == 1
    output = output[0]

    a = C.input_variable((25, 6))
    b = C.layers.AveragePooling(filter_shape=(4, ), strides=(2, ), pad=True)(a)

    n = np.arange(1 * 6 * 25).reshape((1, 6, 25)).astype(np.float32)
    n = np.ascontiguousarray(np.moveaxis(n, 1, -1))

    desired = b.eval({a: n})
    desired = np.squeeze(np.moveaxis(desired, -1, 1))

    # ignore the first and last seq item because SequentialAveragePooling included padding in average calculation
    np.testing.assert_almost_equal(output[1:-1], desired[1:-1])


def test_sequential_concat_pooling():
    a = C.sequence.input_variable((3, 10))
    b = SequentialConcatPooling(filter_shape=(2, 2), strides=2)(a)

    assert b.shape == (6, 5)

    n = [np.random.random((3, 10)).astype(np.float32),
         np.random.random((3, 10)).astype(np.float32),
         np.random.random((3, 10)).astype(np.float32),
         np.random.random((3, 10)).astype(np.float32), ]

    b.eval({a: n})

    b = SequentialConcatPooling(filter_shape=(2, 2), strides=2, pad=False)(a)

    assert b.shape == (6, 5)

    n = [np.random.random((3, 10)).astype(np.float32),
         np.random.random((3, 10)).astype(np.float32),
         np.random.random((3, 10)).astype(np.float32),
         np.random.random((3, 10)).astype(np.float32), ]

    b.eval({a: n})


def test_convolution_2d():
    # --------------------------------------------------------------------------
    # Normal use case
    # --------------------------------------------------------------------------
    image_shape = (3, 32, 32)
    a = C.input_variable(image_shape)
    b = C.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=1, pad=True)(a)
    c = Cx.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=1, pad=True,
                                init=b.W.value, init_bias=b.b.value)(a)

    assert b.shape == c.shape

    n = np.random.random((1, ) + image_shape).astype(np.float32)

    desired = b.eval({a: n})
    actual = c.eval({a: n})

    np.testing.assert_equal(actual, desired)

    image_shape = (3, 32, 32)
    a = C.input_variable(image_shape)
    b = C.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=1, pad=(True, True))(a)
    c = Cx.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=1, pad=(True, True),
                                init=b.W.value, init_bias=b.b.value)(a)

    assert b.shape == c.shape

    n = np.random.random((1,) + image_shape).astype(np.float32)

    desired = b.eval({a: n})
    actual = c.eval({a: n})

    np.testing.assert_equal(actual, desired)

    # --------------------------------------------------------------------------
    # Normal use case, no padding with stride 2
    # --------------------------------------------------------------------------
    image_shape = (3, 32, 32)
    a = C.input_variable(image_shape)
    b = C.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=1, pad=False, strides=2)(a)
    c = Cx.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=1, pad=False, strides=2,
                                init=b.W.value, init_bias=b.b.value)(a)

    assert b.shape == c.shape

    n = np.random.random((1,) + image_shape).astype(np.float32)

    desired = b.eval({a: n})
    actual = c.eval({a: n})

    np.testing.assert_equal(actual, desired)

    # --------------------------------------------------------------------------
    # Normal use case, reduction rank = 0
    # --------------------------------------------------------------------------
    image_shape = (32, 32)
    a = C.input_variable(image_shape)
    b = C.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=0, pad=False, strides=2)(a)
    c = Cx.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=0, pad=False, strides=2,
                                init=b.W.value, init_bias=b.b.value)(a)

    assert b.shape == c.shape

    n = np.random.random((1,) + image_shape).astype(np.float32)

    desired = b.eval({a: n})
    actual = c.eval({a: n})

    np.testing.assert_equal(actual, desired)

    # --------------------------------------------------------------------------
    # Normal use case, reduction rank = 0 with padding
    # --------------------------------------------------------------------------
    image_shape = (32, 32)
    a = C.input_variable(image_shape)
    b = C.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=0, pad=True, strides=2)(a)
    c = Cx.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=0, pad=True, strides=2,
                                init=b.W.value, init_bias=b.b.value)(a)

    assert b.shape == c.shape

    n = np.random.random((1,) + image_shape).astype(np.float32)

    desired = b.eval({a: n})
    actual = c.eval({a: n})

    np.testing.assert_equal(actual, desired)

    # --------------------------------------------------------------------------
    # Normal use case, reduction rank = 0 with pad & stride on one dim
    # --------------------------------------------------------------------------
    image_shape = (32, 32)
    a = C.input_variable(image_shape)
    b = C.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=0, pad=(True, False), strides=(2, 1))(a)
    c = Cx.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=0, pad=(True, False), strides=(2, 1),
                                init=b.W.value, init_bias=b.b.value)(a)

    assert b.shape == c.shape

    n = np.random.random((1,) + image_shape).astype(np.float32)

    desired = b.eval({a: n})
    actual = c.eval({a: n})

    np.testing.assert_equal(actual, desired)

    # --------------------------------------------------------------------------
    # Normal use case, reduction rank = 0 with pad & stride on one dim and no output depth
    # --------------------------------------------------------------------------
    image_shape = (32, 32)
    a = C.input_variable(image_shape)
    b = C.layers.Convolution2D(filter_shape=3, num_filters=None, reduction_rank=0, pad=(True, False), strides=(2, 1))(a)
    c = Cx.layers.Convolution2D(filter_shape=3, num_filters=None, reduction_rank=0, pad=(True, False), strides=(2, 1),
                                init=b.W.value, init_bias=b.b.value)(a)

    assert b.shape == c.shape

    n = np.random.random((1,) + image_shape).astype(np.float32)

    desired = b.eval({a: n})
    actual = c.eval({a: n})

    np.testing.assert_equal(actual, desired)

    # --------------------------------------------------------------------------
    # Image is same size as kernel
    # --------------------------------------------------------------------------
    image_shape = (3, 3, 3)
    a = C.input_variable(image_shape)
    b = C.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=1, pad=True)(a)
    c = Cx.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=1, pad=True,
                                init=b.W.value, init_bias=b.b.value)(a)

    assert b.shape == c.shape

    n = np.random.random((1, ) + image_shape).astype(np.float32)

    desired = b.eval({a: n})
    actual = c.eval({a: n})

    np.testing.assert_equal(actual, desired)

    # --------------------------------------------------------------------------
    # Image is same size as kernel, no padding
    # --------------------------------------------------------------------------
    image_shape = (3, 3, 3)
    a = C.input_variable(image_shape)
    b = C.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=1, pad=False)(a)
    c = Cx.layers.Convolution2D(filter_shape=3, num_filters=16, reduction_rank=1, pad=False,
                                init=b.W.value, init_bias=b.b.value)(a)

    assert b.shape == c.shape

    n = np.random.random((1,) + image_shape).astype(np.float32)

    desired = b.eval({a: n})
    actual = c.eval({a: n})

    np.testing.assert_equal(actual, desired)

    # --------------------------------------------------------------------------
    # Check kernel initialised same shape
    # --------------------------------------------------------------------------
    image_shape = (32, 32)
    a = C.input_variable(image_shape)
    b = C.layers.Convolution2D(filter_shape=(3, 2), num_filters=None, reduction_rank=0, pad=(True, False), strides=(2, 1))(a)
    c = Cx.layers.Convolution2D(filter_shape=(3, 2), num_filters=None, reduction_rank=0, pad=(True, False), strides=(2, 1))(a)

    assert b.shape == c.shape
    assert b.W.shape == c.W.shape
    assert b.b.shape == c.b.shape

    image_shape = (3, 32, 32)
    a = C.input_variable(image_shape)
    b = C.layers.Convolution2D(filter_shape=(3, 4), num_filters=None, reduction_rank=1, pad=(True, False), strides=(2, 1))(a)
    c = Cx.layers.Convolution2D(filter_shape=(3, 4), num_filters=None, reduction_rank=1, pad=(True, False), strides=(2, 1))(a)

    assert b.shape == c.shape
    assert b.W.shape == c.W.shape
    assert b.b.shape == c.b.shape


def test_sequential_convolution():
    # --------------------------------------------------------------------------
    # Normal use case - image check kernel initialisation shape
    # --------------------------------------------------------------------------
    image_shape = (3, 32)  # rgb image of variable width
    a = C.sequence.input_variable(image_shape)
    b = C.layers.SequentialConvolution(filter_shape=(3, 3), num_filters=16, reduction_rank=1, pad=True)(a)
    c = Cx.layers.SequentialConvolution(filter_shape=(3, 3), num_filters=16, reduction_rank=1, pad=True)(a)

    assert b.shape == c.shape
    assert b.W.shape == c.W.shape
    assert b.b.shape == c.b.shape

    image_shape = (32, )  # black and white image of variable width
    a = C.sequence.input_variable(image_shape)
    b = C.layers.SequentialConvolution(filter_shape=(2, 2), num_filters=16, reduction_rank=0, pad=True)(a)
    c = Cx.layers.SequentialConvolution(filter_shape=(2, 2), num_filters=16, reduction_rank=0, pad=True)(a)

    assert b.shape == c.shape
    assert b.W.shape == c.W.shape
    assert b.b.shape == c.b.shape

    image_shape = (32,)  # text vector of variable sequence length
    a = C.sequence.input_variable(image_shape)
    b = C.layers.SequentialConvolution(filter_shape=(2, 2), num_filters=16, reduction_rank=1, pad=True)(a)
    c = Cx.layers.SequentialConvolution(filter_shape=(2, 2), num_filters=16, reduction_rank=1, pad=True)(a)

    assert b.shape == c.shape
    assert b.W.shape == c.W.shape
    assert b.b.shape == c.b.shape

    # --------------------------------------------------------------------------
    # Normal use case - image
    # --------------------------------------------------------------------------
    kernel_init = C.glorot_normal(seed=5)
    image_shape = (3, 32)
    width = 40
    a = C.sequence.input_variable(image_shape)
    b = C.layers.SequentialConvolution(filter_shape=(3, 3), num_filters=16, reduction_rank=1, pad=True, init=kernel_init)(a)
    c = Cx.layers.SequentialConvolution(filter_shape=(3, 3), num_filters=16, reduction_rank=1, pad=True, init=kernel_init)(a)

    assert b.shape == c.shape

    n = np.random.random((1, width) + image_shape).astype(np.float32)

    desired = b.eval({a: n})
    actual = c.eval({a: n})

    if isinstance(desired, list) and len(desired) == 1:
        desired = desired[0]

    if isinstance(actual, list) and len(actual) == 1:
        actual = actual[0]

    np.testing.assert_equal(actual, desired)

    # --------------------------------------------------------------------------
    # Normal use case - image, mix padding across axis
    # --------------------------------------------------------------------------
    kernel_init = C.glorot_normal(seed=5)
    image_shape = (3, 32)
    width = 40
    a = C.sequence.input_variable(image_shape)
    b = C.layers.SequentialConvolution(filter_shape=(3, 3), num_filters=16, reduction_rank=1, pad=(False, True), init=kernel_init)(a)
    c = Cx.layers.SequentialConvolution(filter_shape=(3, 3), num_filters=16, reduction_rank=1, pad=(False, True), init=kernel_init)(a)

    assert b.shape == c.shape

    n = np.random.random((1, width) + image_shape).astype(np.float32)

    desired = b.eval({a: n})
    actual = c.eval({a: n})

    if isinstance(desired, list) and len(desired) == 1:
        desired = desired[0]

    if isinstance(actual, list) and len(actual) == 1:
        actual = actual[0]

    np.testing.assert_equal(actual, desired)

    # --------------------------------------------------------------------------
    # Normal use case - vector data for qrnn implementation
    # --------------------------------------------------------------------------
    kernel_init = C.glorot_normal(seed=5)
    image_shape = (25,)
    width = 40
    a = C.sequence.input_variable(image_shape)
    b = C.layers.SequentialConvolution(filter_shape=(2, ), num_filters=16, reduction_rank=1, pad=True, init=kernel_init)(a)
    c = Cx.layers.SequentialConvolution(filter_shape=(2,), num_filters=16, reduction_rank=1, pad=True, init=kernel_init)(a)

    assert b.shape == c.shape

    n = np.random.random((1, width) + image_shape).astype(np.float32)

    desired = b.eval({a: n})
    actual = c.eval({a: n})

    if isinstance(desired, list) and len(desired) == 1:
        desired = desired[0]

    if isinstance(actual, list) and len(actual) == 1:
        actual = actual[0]

    np.testing.assert_equal(actual, desired)

    # --------------------------------------------------------------------------
    # Normal use case - B&W image
    # --------------------------------------------------------------------------
    kernel_init = C.glorot_normal(seed=5)
    image_shape = (32, )
    width = 40
    a = C.sequence.input_variable(image_shape)
    b = C.layers.SequentialConvolution(filter_shape=(3, 3), num_filters=16, reduction_rank=0, pad=True, init=kernel_init)(a)
    c = Cx.layers.SequentialConvolution(filter_shape=(3, 3), num_filters=16, reduction_rank=0, pad=True, init=kernel_init)(a)

    assert b.shape == c.shape

    n = np.random.random((1, width) + image_shape).astype(np.float32)

    desired = b.eval({a: n})
    actual = c.eval({a: n})

    if isinstance(desired, list) and len(desired) == 1:
        desired = desired[0]

    if isinstance(actual, list) and len(actual) == 1:
        actual = actual[0]

    np.testing.assert_equal(actual, desired)

    image_shape = (32,)
    width = 40
    a = C.sequence.input_variable(image_shape)
    b = C.layers.SequentialConvolution(filter_shape=(3, 3), num_filters=16, reduction_rank=0, pad=True)(a)
    c = Cx.layers.SequentialConvolution(filter_shape=(3, 3), num_filters=16, reduction_rank=0, pad=True, init=b.W.value)(a)

    assert b.shape == c.shape

    n = np.random.random((1, width) + image_shape).astype(np.float32)

    desired = b.eval({a: n})
    actual = c.eval({a: n})

    if isinstance(desired, list) and len(desired) == 1:
        desired = desired[0]

    if isinstance(actual, list) and len(actual) == 1:
        actual = actual[0]

    np.testing.assert_equal(actual, desired)


def test_vfsmn():
    in_dim = 10
    hidden_dim = 100
    num_past_context = 3
    num_future_context = 0
    a = C.sequence.input_variable(in_dim)
    b = vFSMN(hidden_dim, C.relu, num_past_context, num_future_context)(a)

    assert b.shape == (hidden_dim,)
    assert b.b.shape == (hidden_dim,)
    assert b.a.shape == (num_past_context + num_future_context + 1, in_dim)
    assert b.H.shape == (in_dim, hidden_dim)
    assert b.W.shape == (in_dim, hidden_dim)

    n = [np.random.random((15, in_dim)).astype(np.float32),
         np.random.random((7, in_dim)).astype(np.float32),
         np.random.random((20, in_dim)).astype(np.float32), ]

    b.eval({a: n})

    in_dim = 10
    hidden_dim = 20
    num_past_context = 3
    num_future_context = 3
    a = C.sequence.input_variable(in_dim)
    b = vFSMN(hidden_dim, C.relu, num_past_context, num_future_context)(a)

    assert b.shape == (hidden_dim,)
    assert b.b.shape == (hidden_dim,)
    assert b.a.shape == (num_past_context + num_future_context + 1, in_dim)
    assert b.H.shape == (in_dim, hidden_dim)
    assert b.W.shape == (in_dim, hidden_dim)

    n = [np.random.random((15, in_dim)).astype(np.float32),
         np.random.random((7, in_dim)).astype(np.float32),
         np.random.random((20, in_dim)).astype(np.float32), ]

    b.eval({a: n})


def test_cfsmn():
    in_dim = 50
    hidden_dim = 100
    proj_dim = 10
    num_past_context = 3
    num_future_context = 0
    a = C.sequence.input_variable(in_dim)
    b = cFSMN(hidden_dim, proj_dim, C.relu, num_past_context, num_future_context)(a)

    assert b.shape == (hidden_dim,)
    assert b.b.shape == (proj_dim,)
    assert b.bb.shape == (hidden_dim,)
    assert b.a.shape == (num_past_context + num_future_context + 1, proj_dim)
    assert b.H.shape == (proj_dim, hidden_dim)
    assert b.W.shape == (in_dim, proj_dim)

    n = [np.random.random((15, in_dim)).astype(np.float32),
         np.random.random((7, in_dim)).astype(np.float32),
         np.random.random((20, in_dim)).astype(np.float32), ]

    b.eval({a: n})

    in_dim = 10
    hidden_dim = 20
    num_past_context = 3
    num_future_context = 3
    a = C.sequence.input_variable(in_dim)
    b = cFSMN(hidden_dim, proj_dim, C.relu, num_past_context, num_future_context)(a)

    assert b.shape == (hidden_dim,)
    assert b.b.shape == (proj_dim,)
    assert b.bb.shape == (hidden_dim,)
    assert b.a.shape == (num_past_context + num_future_context + 1, proj_dim)
    assert b.H.shape == (proj_dim, hidden_dim)
    assert b.W.shape == (in_dim, proj_dim)

    n = [np.random.random((15, in_dim)).astype(np.float32),
         np.random.random((7, in_dim)).astype(np.float32),
         np.random.random((20, in_dim)).astype(np.float32), ]

    b.eval({a: n})


def test_sequential_dense():
    # ====================================================
    # window = 2 stride = 1
    # ====================================================
    in_dim = 5
    out_dim = 10
    a = C.sequence.input_variable(in_dim)
    b = Cx.layers.SequentialDense(out_dim, window=2, stride=1, causal=False)(a)

    assert b.shape == (out_dim, )

    n = [np.random.random((15, in_dim)).astype(np.float32),
         np.random.random((7, in_dim)).astype(np.float32),
         np.random.random((20, in_dim)).astype(np.float32), ]

    results = b.eval({a: n})

    for r, nn in zip(results, n):
        assert r.shape == (nn.shape[0], out_dim)

    # ====================================================
    # window = 2 stride = 2
    # ====================================================
    in_dim = 5
    out_dim = 10
    a = C.sequence.input_variable(in_dim)
    b = Cx.layers.SequentialDense(out_dim, window=2, stride=2, causal=False)(a)

    assert b.shape == (out_dim,)

    n = [np.random.random((15, in_dim)).astype(np.float32),
         np.random.random((7, in_dim)).astype(np.float32),
         np.random.random((20, in_dim)).astype(np.float32), ]

    results = b.eval({a: n})

    for r, nn in zip(results, n):
        assert r.shape == (math.ceil(nn.shape[0] / 2), out_dim)

    # ====================================================
    # window = 3 stride = 3 causal = True
    # ====================================================
    in_dim = 5
    out_dim = 10
    stride = 3
    a = C.sequence.input_variable(in_dim)
    b = Cx.layers.SequentialDense(out_dim, window=3, stride=stride, causal=True)(a)

    assert b.shape == (out_dim,)

    n = [np.random.random((15, in_dim)).astype(np.float32),
         np.random.random((7, in_dim)).astype(np.float32),
         np.random.random((20, in_dim)).astype(np.float32), ]

    results = b.eval({a: n})

    for r, nn in zip(results, n):
        assert r.shape == (math.ceil(nn.shape[0] / stride), out_dim)


def test_filtered_responsed_normalization_layer():
    a = C.input_variable((3, 32, 32))
    b = Cx.layers.FilterResponseNormalization()(a)

    assert b.shape == (3, 32, 32)

    n = np.random.random((10, 3, 32, 32)).astype(np.float32)
    results = b.eval({a: n})

    a = C.sequence.input_variable((3, 32))
    b = Cx.layers.FilterResponseNormalization(num_static_spatial_axes=1, seq_axis_is_spatial=True)(a)

    assert b.shape == (3, 32)

    n = [np.random.random((10, 3, 32)).astype(np.float32),
         np.random.random((7, 3, 32)).astype(np.float32), ]

    results = b.eval({a: n})


def test_boom_layer():
    output_dim = 32
    expansion_factor = 4

    a = C.input_variable(20)
    b = Cx.layers.Boom(output_dim, expansion_factor)(a)

    assert b.shape == (output_dim, )

    n = np.random.random((10, 20)).astype(np.float32)
    results = b.eval({a: n})


def test_se_block():
    a = C.input_variable((8, 32, 16))
    b = Cx.layers.SEBlock(8, r=2)(a)

    assert b.shape == (8, 32, 16)

    n = np.random.random((10, 8, 32, 16)).astype(np.float32)
    b.eval({a: n})


def test_sequence_se_block():
    a = C.sequence.input_variable((32, 24))
    b = Cx.layers.SequenceSEBlock(32, activation=Cx.mish)(a)

    assert b.shape == (32, 24)

    n = [np.random.random((16, 32, 24)).astype(np.float32),
         np.random.random((7, 32, 24)).astype(np.float32), ]
    b.eval({a: n})
