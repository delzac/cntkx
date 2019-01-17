import cntk as C
from cntkx.layers.models import Transformer, TransformerDecoder, TransformerEncoder, MultiHeadAttention
from cntkx.layers.models import MultiHeadAttentionBlock, TransformerEncoderBlock, TransformerDecoderBlock
from cntkx.layers.models import ScaledDotProductAttention
import numpy as np
import pytest


def test_scaled_dot_product_attention1():
    """ check default works """
    s1, s2 = 4, 2
    a = C.sequence.input_variable(5)
    b = ScaledDotProductAttention()(a, a, a)

    assert b.shape == (5, ), "output should be a sequence and dimension should not change"

    n1 = np.random.random((s1, 5)).astype(np.float32)
    n2 = np.random.random((s2, 5)).astype(np.float32)

    results = b.eval({a: [n1, n2]})
    assert results[1].shape == n2.shape, f"Wrong expected shape {results[1].shape} != {n2.shape}"
    assert results[0].shape == n1.shape, f"Wrong expected shape {results[0].shape} != {n1.shape}"


def test_scaled_dot_product_attention2():
    """ returns a sequence while not peeking on future values """
    s1, s2 = 4, 2
    a = C.sequence.input_variable(5)
    b = ScaledDotProductAttention(obey_sequence_order=True, max_seq_len=100)(a, a, a)

    assert b.shape == (5, ), "output should be a sequence and dimension should not change"

    n1 = np.random.random((s1, 5)).astype(np.float32)
    n2 = np.random.random((s2, 5)).astype(np.float32)

    results = b.eval({a: [n1, n2]})
    assert results[1].shape == n2.shape, f"Wrong expected shape {results[1].shape} != {n2.shape}"
    assert results[0].shape == n1.shape, f"Wrong expected shape {results[0].shape} != {n1.shape}"


def test_scaled_dot_product_attention3():
    """ query and key-value musts have same dimensions """
    query = C.sequence.input_variable(5)
    keyvalue = C.sequence.input_variable(20)

    with pytest.raises(Exception):
        b = ScaledDotProductAttention()(query, keyvalue, keyvalue)


def test_multi_head_attention1():
    """ default settings: output as unpacked sequence tensor """
    a = C.sequence.input_variable(5)
    multihead1 = MultiHeadAttention(num_heads=6, model_dim=30)
    multihead2 = MultiHeadAttention(num_heads=6, model_dim=60)

    attended = multihead1(a, a, a)
    assert attended.shape == (30, )

    attended = multihead2(attended, attended, attended)
    assert attended.shape == (60, )

    n = [np.random.random((2, 5)).astype(np.float32),
         np.random.random((4, 5)).astype(np.float32),
         np.random.random((6, 5)).astype(np.float32)]

    results = attended.eval({a: n})
    np.testing.assert_equal(results[0][2:], 0)
    np.testing.assert_equal(results[1][4:], 0)

    with pytest.raises(Exception):
        np.testing.assert_equal(results[1][2:], 0)
        np.testing.assert_equal(results[2][4:], 0)


def test_multi_head_attention2():
    """ no peeking into future values """
    a = C.sequence.input_variable(5)
    multihead1 = MultiHeadAttention(num_heads=6, model_dim=30, obey_sequence_order=True,
                                    max_seq_len=10)

    attended = multihead1(a, a, a)
    assert attended.shape == (30,)

    n = np.random.random((2, 3, 5)).astype(np.float32)
    attended.eval({a: n})

    # Exceed max_seq_length should raise error
    n = np.random.random((2, 11, 5)).astype(np.float32)
    with pytest.raises(Exception):
        attended.eval({a: n})


def test_multi_head_attention3():
    """ different dimension in between query and key-value pair """
    seq1 = C.Axis.new_unique_dynamic_axis('seq1')
    seq2 = C.Axis.new_unique_dynamic_axis('seq2')

    a = C.sequence.input_variable(30, sequence_axis=seq1)
    b = C.sequence.input_variable(10, sequence_axis=seq2)

    multihead1 = MultiHeadAttention(num_heads=6, model_dim=30, obey_sequence_order=True,
                                    max_seq_len=10)

    attended = multihead1(b, a, a)
    assert attended.shape == (30,)

    n = np.random.random((2, 3, 30)).astype(np.float32)
    m = np.random.random((2, 5, 10)).astype(np.float32)
    attended.eval({a: n, b: m})


def test_multi_head_attention_w_recurrence_lstm():
    """ combined multi head attention with lstm recurrence

    encoding seq and decoding seq has different seq length and different dimensions

    Multi head attention also has different dimension with lstm and decoding seq dim.

    """
    seq1 = C.Axis.new_unique_dynamic_axis('seq1')
    seq2 = C.Axis.new_unique_dynamic_axis('seq2')

    e = C.sequence.input_variable(10, sequence_axis=seq1)
    a = C.sequence.input_variable(20, sequence_axis=seq2)
    mha = MultiHeadAttention(num_heads=5, model_dim=20)
    lstm = C.layers.LSTM(30)

    @C.Function
    def decoder(encoded, target):

        @C.Function
        def lstm_w_attention(h, c, x):
            attended = mha(h, encoded, encoded)

            xx = C.splice(attended, x)
            return lstm(h, c, xx)

        output = C.layers.Recurrence(lstm_w_attention)(target)

        return output

    decoded = decoder(e, a)

    m = np.random.random((3, 10, 10)).astype(np.float32)

    n = [np.random.random((2, 20)).astype(np.float32),
         np.random.random((4, 20)).astype(np.float32),
         np.random.random((6, 20)).astype(np.float32)]

    results = decoded.eval({e: m, a: n})


def test_multi_head_attention_block1():
    """ default settings """
    a = C.sequence.input_variable(20)
    multihead1 = MultiHeadAttentionBlock(num_heads=5, model_dim=20)

    attended = multihead1(a, a, a)
    assert attended.shape == (20, )

    n = [np.random.random((2, 20)).astype(np.float32),
         np.random.random((4, 20)).astype(np.float32),
         np.random.random((6, 20)).astype(np.float32)]

    results = attended.eval({a: n})


def test_multi_head_attention_block2():
    """ no peek into future """
    a = C.sequence.input_variable(10)

    multihead1 = MultiHeadAttentionBlock(num_heads=2, model_dim=10, obey_sequence_order=True, max_seq_len=100)

    attended = multihead1(a, a, a)
    assert attended.shape == (10, )

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = attended.eval({a: n})
    np.testing.assert_equal(results[0][2:], 0)
    np.testing.assert_equal(results[1][4:], 0)


def test_multi_head_attention_block3():
    """ difference input dimension and model dim should raise exception due to resnet connection """
    a = C.sequence.input_variable(20)
    multihead1 = MultiHeadAttentionBlock(num_heads=5, model_dim=40)

    with pytest.raises(Exception):
        attended = multihead1(a, a, a)


def test_transformer_encoder_block1a():
    """ Default settings: input is seq output is not seq """
    a = C.sequence.input_variable(10)
    encoder_block = TransformerEncoderBlock(2, 10)
    attended = encoder_block(a)

    assert attended.shape == (10, )

    n = [np.random.random((3, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = attended.eval({a: n})


def test_transformer_decoder_block1():
    """ default settings: encoder block output feed into decoder block """
    a = C.sequence.input_variable(10)
    b = C.sequence.input_variable(10)

    encoder_block = TransformerEncoderBlock(num_heads=2, model_dim=10)
    decoder_block = TransformerDecoderBlock(num_heads=2, model_dim=10)

    encoded = encoder_block(a)
    decoded = decoder_block(encoded, b)

    assert decoded.shape == (10, )

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]
    m = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = decoded.eval({a: n, b: m})


def test_transformer_decoder_block2():
    """ default settings: encoder block output feed into decoder block """
    a = C.sequence.input_variable(10)
    b = C.sequence.input_variable(10)

    encoder_block = TransformerEncoderBlock(num_heads=2, model_dim=10)
    decoder_block = TransformerDecoderBlock(num_heads=2, model_dim=10, obey_sequence_order=True, max_seq_len=100)

    encoded = encoder_block(a)
    decoded = decoder_block(encoded, b)

    assert decoded.shape == (10, )

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]
    m = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = decoded.eval({a: n, b: m})


def test_transformer_decoder_block3():
    """ Typical use case: encoder feed into decoder with multi layer """
    a = C.sequence.input_variable(10)
    b = C.sequence.input_variable(10)

    encoder_block = TransformerEncoderBlock(num_heads=2, model_dim=10)
    decoder_block1 = TransformerDecoderBlock(num_heads=2, model_dim=10)
    decoder_block2 = TransformerDecoderBlock(num_heads=2, model_dim=10)

    encoded = encoder_block(a)
    decoded = decoder_block1(encoded, b)
    decoded = decoder_block2(encoded, decoded)

    assert decoded.shape == (10, )

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]
    m = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = decoded.eval({a: n, b: m})


def test_transformer_encoder1a():
    """ multi-layers encoders """
    a = C.sequence.input_variable(10)
    encoder = TransformerEncoder(5, 2, 10)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = encoded.eval({a: n})

    encoder = TransformerEncoder(2, 2, 10)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    results = encoded.eval({a: n})

    encoder = TransformerEncoder(1, 2, 10)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    results = encoded.eval({a: n})


def test_transformer_encoder1c():
    """ No peeking and multi layers encoder """
    a = C.sequence.input_variable(10)
    encoder = TransformerEncoder(5, 2, 10, obey_sequence_order=True, max_seq_len=100)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = encoded.eval({a: n})

    encoder = TransformerEncoder(2, 2, 10, obey_sequence_order=True, max_seq_len=100)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    results = encoded.eval({a: n})

    encoder = TransformerEncoder(1, 2, 10, obey_sequence_order=True, max_seq_len=100)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    results = encoded.eval({a: n})


def test_transformer_decoder1():
    """ default setup: 5-1 layers of decoders """
    seq1 = C.Axis.new_unique_dynamic_axis('seq1')
    seq2 = C.Axis.new_unique_dynamic_axis('seq2')

    a = C.sequence.input_variable(10, sequence_axis=seq1)
    b = C.sequence.input_variable(10, sequence_axis=seq2)

    decoder = TransformerDecoder(5, 2, 10)

    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    m = np.random.random((4, 8, 10)).astype(np.float32)

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(3, 2, 10)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(2, 2, 10)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(1, 2, 10)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})


def test_transformer_decoder2():
    """ output as sequence with no peeking at future values """
    seq1 = C.Axis.new_unique_dynamic_axis('seq1')
    seq2 = C.Axis.new_unique_dynamic_axis('seq2')

    a = C.sequence.input_variable(10, sequence_axis=seq1)
    b = C.sequence.input_variable(10, sequence_axis=seq2)

    decoder = TransformerDecoder(5, 2, 10, obey_sequence_order=True, max_seq_len=100)

    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    m = np.random.random((4, 8, 10)).astype(np.float32)

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(3, 2, 10, obey_sequence_order=True, max_seq_len=100)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(2, 2, 10, obey_sequence_order=True, max_seq_len=100)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(1, 2, 10, obey_sequence_order=True, max_seq_len=100)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})


def test_transformer_decoder3():
    """
    Different dimensions between encoded and decoder model is allowed as
    encoded will be cast to model_dim of decoder.
    """
    seq1 = C.Axis.new_unique_dynamic_axis('seq1')
    seq2 = C.Axis.new_unique_dynamic_axis('seq2')

    a = C.sequence.input_variable(30, sequence_axis=seq1)
    b = C.sequence.input_variable(10, sequence_axis=seq2)

    decoder = TransformerDecoder(5, 2, 10, obey_sequence_order=True, max_seq_len=100)

    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    m = np.random.random((4, 8, 30)).astype(np.float32)

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = decoded.eval({a: m, b: n})


def test_transformer1():
    """ default configuration of using transformer """
    a = C.sequence.input_variable(10)
    b = C.sequence.input_variable(10)

    transformer = Transformer(num_encoder_blocks=3, num_decoder_blocks=3, num_heads_encoder=2, num_heads_decoder=2,
                              encoder_model_dim=10, decoder_model_dim=10, encoder_obey_sequence_order=False,
                              decoder_obey_sequence_order=True, max_seq_len_encoder=100, max_seq_len_decoder=100)

    prediction = transformer(a, b)

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = prediction.eval({a: n, b: n})


def test_transformer2():
    """ different sequence length and dimension between encoder and decoder """
    seq1 = C.Axis.new_unique_dynamic_axis('seq1')
    seq2 = C.Axis.new_unique_dynamic_axis('seq2')

    encoding = C.sequence.input_variable(10, sequence_axis=seq1)
    decoding = C.sequence.input_variable(30, sequence_axis=seq2)

    transformer = Transformer(num_encoder_blocks=3, num_decoder_blocks=3, num_heads_encoder=2, num_heads_decoder=2,
                              encoder_model_dim=10, decoder_model_dim=30, encoder_obey_sequence_order=False,
                              decoder_obey_sequence_order=True, max_seq_len_encoder=100, max_seq_len_decoder=100)

    prediction = transformer(encoding, decoding)

    n = [np.random.random((4, 30)).astype(np.float32),
         np.random.random((8, 30)).astype(np.float32),
         np.random.random((16, 30)).astype(np.float32),
         np.random.random((20, 30)).astype(np.float32)]

    m = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32),]

    results = prediction.eval({encoding: m, decoding: n})
