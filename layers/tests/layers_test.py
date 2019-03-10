import cntk as C
from cntkx.layers import QRNN, SinusoidalPositionalEmbedding, SpatialPyramidPooling, GatedLinearUnit
from cntkx.layers import VariationalDropout, WeightDroppedLSTM, BertEmbeddings, PositionalEmbedding
from cntkx.layers import PreTrainedBertEmbeddings, PositionwiseFeedForward
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


def test_variational_dropout():
    a = C.sequence.input_variable(10)
    b = VariationalDropout(0.1)(a)

    assert b.shape == a.shape

    n1 = np.random.random((3, 10)).astype(np.float32)
    n2 = np.random.random((6, 10)).astype(np.float32)

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

    embeddings = PreTrainedBertEmbeddings(filepath_to_tf_bert_model, 0.1)
    b = embeddings(text_tensor, token_type_tensor)

    assert b.shape == (768,)

    n1 = np.random.random((3, 30522)).astype(np.float32)
    n2 = np.random.random((6, 30522)).astype(np.float32)

    m1 = np.random.random((3, 2)).astype(np.float32)
    m2 = np.random.random((6, 2)).astype(np.float32)
    b.eval({text_tensor: [n1, n2], token_type_tensor: [m1, m2]})

    embed = getattr(b, "bert/embeddings/word_embeddings")
    assert embed.shape == (768,)

    embed = getattr(b, "bert/embeddings/position_embeddings")
    assert embed.parameters[0].shape == (512, 768)

    embed = getattr(b, "bert/embeddings/token_type_embeddings")
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
