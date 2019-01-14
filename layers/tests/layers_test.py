import cntk as C
from cntkx.layers import QRNN, MultiheadAttention, MultiHeadAttentionBlock, TransformerEncoderBlock
from cntkx.layers import TransformerDecoderBlock, SinusoidalPositionalEmbedding
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
    multihead1 = MultiheadAttention(num_heads=6, model_dim=30)
    multihead2 = MultiheadAttention(num_heads=6, model_dim=60, map_ranks=1)

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
    multihead1 = MultiheadAttention(num_heads=6, model_dim=30)
    multihead2 = MultiheadAttention(num_heads=6, model_dim=60, map_ranks=1)

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
    multihead1 = MultiheadAttention(num_heads=6, model_dim=30, output_as_seq=True)
    multihead2 = MultiheadAttention(num_heads=6, model_dim=60, output_as_seq=True)

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
    multihead1 = MultiheadAttention(num_heads=6, model_dim=30, obey_sequence_order=True,
                                    max_seq_len=10, output_as_seq=True)

    attended = multihead1(a, a, a)
    assert attended.shape == (30,)

    n = np.random.random((2, 3, 5)).astype(np.float32)
    attended.eval({a: n})

    n = np.random.random((2, 11, 5)).astype(np.float32)
    with pytest.raises(Exception):
        attended.eval({a: n})


def test_multi_head_attention_block1a():
    """ default settings: output as unpacked sequence tensor """
    a = C.sequence.input_variable(10)
    multihead1 = MultiHeadAttentionBlock(num_heads=2, model_dim=10)

    attended = multihead1(a, a, a, None)
    assert attended.shape == (-3, 10)

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = attended.eval({a: n})


def test_multi_head_attention_block1b():
    """ default settings: output as unpacked sequence tensor """
    a = C.sequence.input_variable(10)
    multihead1 = MultiHeadAttentionBlock(num_heads=2, model_dim=10, output_as_seq=True)

    attended = multihead1(a, a, a, None)
    assert attended.shape == (10, )

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = attended.eval({a: n})


def test_multi_head_attention_block1c():
    """ default settings: output as unpacked sequence tensor """
    a = C.input_variable((-3, 10))
    b = C.sequence.input_variable(10)

    multihead1 = MultiHeadAttentionBlock(num_heads=2, model_dim=10, map_ranks=1)

    attended = multihead1(a, a, a, b)
    assert attended.shape == (-3, 10)

    o = np.random.random((3, 6, 10)).astype(np.float32)
    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = attended.eval({a: o, b: n})
    with pytest.raises(Exception):
        np.testing.assert_equal(results[0][2:], 0)
        np.testing.assert_equal(results[1][4:], 0)


def test_transformer_encoder_block1a():
    """ Default settings: input is seq output is not seq """
    a = C.sequence.input_variable(10)
    encoder_block = TransformerEncoderBlock(2, 10)
    attended = encoder_block(a, None)

    assert attended.shape == (-3, 10)

    n = [np.random.random((3, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = attended.eval({a: n})


def test_transformer_encoder_block1b():
    """ Default settings: input is seq output is seq """
    a = C.sequence.input_variable(10)
    encoder_block = TransformerEncoderBlock(2, 10, output_as_seq=True)
    attended = encoder_block(a, None)

    assert attended.shape == (10,)

    n = [np.random.random((3, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = attended.eval({a: n})


def test_transformer_encoder_block1c():
    """ Default settings: input is not seq output is not seq """
    a = C.input_variable((-3, 10))
    encoder_block = TransformerEncoderBlock(2, 10, map_rank=1)
    attended = encoder_block(a, None)

    assert attended.shape == (-3, 10)

    n = np.random.random((2, 6, 10)).astype(np.float32)
    results = attended.eval({a: n})


def test_transformer_encoder_block1d():
    """ Default settings: input is not seq output is seq """
    a = C.input_variable((-3, 10))
    b = C.sequence.input_variable(10)
    encoder_block = TransformerEncoderBlock(2, 10, map_rank=1, output_as_seq=True)
    attended = encoder_block(a, b)

    assert attended.shape == (10,)

    n = np.random.random((2, 6, 10)).astype(np.float32)
    m = [np.random.random((3, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = attended.eval({a: n, b: m})


def test_transformer_decoder_block1():
    """ default settings: encoded is not seq and decoder expects it correctly """
    a = C.sequence.input_variable(10)
    b = C.sequence.input_variable(10)

    encoder_block = TransformerEncoderBlock(num_heads=2, model_dim=10)
    decoder_block = TransformerDecoderBlock(num_heads=2, model_dim=10, is_encoded_seq=False, map_rank=None)

    encoded = encoder_block(a)
    decoded = decoder_block(encoded, b, None)

    assert decoded.shape == (-3, 10)

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]
    m = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = decoded.eval({a: n, b: m})


def test_transformer_decoder_block2():
    """ encoded is seq and decoded expects it correctly """
    a = C.sequence.input_variable(10)
    b = C.sequence.input_variable(10)

    encoder_block = TransformerEncoderBlock(num_heads=2, model_dim=10, output_as_seq=True)
    decoder_block = TransformerDecoderBlock(num_heads=2, model_dim=10, is_encoded_seq=True, map_rank=None)

    encoded = encoder_block(a)
    decoded = decoder_block(encoded, b, None)

    assert decoded.shape == (-3, 10)

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]
    m = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = decoded.eval({a: n, b: m})


def test_transformer_decoder_block2a():
    """ encoded is seq and decoder doesn't expect it """
    a = C.sequence.input_variable(10)
    b = C.sequence.input_variable(10)

    encoder_block = TransformerEncoderBlock(num_heads=2, model_dim=10, output_as_seq=True)
    decoder_block = TransformerDecoderBlock(num_heads=2, model_dim=10, is_encoded_seq=False, map_rank=None)

    encoded = encoder_block(a)
    with pytest.raises(Exception):
        decoded = decoder_block(encoded, b, None)


def test_transformer_decoder_block3():
    """ Typical use case: encoder outputs non-seq, decoder has multi layer """
    a = C.sequence.input_variable(10)
    b = C.sequence.input_variable(10)

    encoder_block = TransformerEncoderBlock(num_heads=2, model_dim=10)
    decoder_block1 = TransformerDecoderBlock(num_heads=2, model_dim=10, is_encoded_seq=False, map_rank=None)
    decoder_block2 = TransformerDecoderBlock(num_heads=2, model_dim=10, is_encoded_seq=False, map_rank=1)

    encoded = encoder_block(a)
    decoded = decoder_block1(encoded, b, None)
    decoded = decoder_block2(encoded, decoded, b)

    assert decoded.shape == (-3, 10)

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]
    m = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = decoded.eval({a: n, b: m})


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
