import cntk as C
from cntkx.layers.models import Transformer, TransformerDecoder, TransformerEncoder, MultiHeadAttention
from cntkx.layers.models import MultiHeadAttentionBlock, TransformerEncoderBlock, TransformerDecoderBlock
from cntkx.layers.models import ScaledDotProductAttention, GaussianWindowAttention, PreTrainedBertEncoder
from cntkx.layers.models import PreTrainedBertModel
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


def test_scaled_dot_product_attention4():
    """ value can be of a completely different dimensions as the query or key """
    s1, s2 = 4, 2

    query = C.sequence.input_variable(5)
    key = C.sequence.input_variable(5)
    value = C.sequence.input_variable(7)

    b = ScaledDotProductAttention()(query, key, value)

    n1 = np.random.random((s1, 5)).astype(np.float32)
    n2 = np.random.random((s2, 5)).astype(np.float32)

    m1 = np.random.random((s1, 7)).astype(np.float32)
    m2 = np.random.random((s2, 7)).astype(np.float32)

    results = b.eval({query: [n1, n2], key: [n1, n2], value: [m1, m2]})


def test_scaled_dot_product_attention5():
    """
    value different dimension to query and key, AND
    query and key can have different sequence length, AND
    key and value must have same sequence length
    """
    s1, s2 = 4, 2
    s3, s4 = 5, 10

    seq1 = C.Axis.new_unique_dynamic_axis('seq1')
    seq2 = C.Axis.new_unique_dynamic_axis('seq2')

    query = C.sequence.input_variable(5, sequence_axis=seq1)
    key = C.sequence.input_variable(5, sequence_axis=seq2)
    value = C.sequence.input_variable(7, sequence_axis=seq2)

    b = ScaledDotProductAttention()(query, key, value)

    q1 = np.random.random((s3, 5)).astype(np.float32)
    q2 = np.random.random((s4, 5)).astype(np.float32)

    k1 = np.random.random((s1, 5)).astype(np.float32)
    k2 = np.random.random((s2, 5)).astype(np.float32)

    v1 = np.random.random((s1, 7)).astype(np.float32)
    v2 = np.random.random((s2, 7)).astype(np.float32)

    results = b.eval({query: [q1, q2], key: [k1, k2], value: [v1, v2]})


def test_scaled_dot_product_attention6():
    """
    key and value must have same sequence length
    """
    s1, s2 = 4, 2
    s3, s4 = 5, 10

    seq1 = C.Axis.new_unique_dynamic_axis('seq1')
    seq2 = C.Axis.new_unique_dynamic_axis('seq2')

    query = C.sequence.input_variable(5, sequence_axis=seq1)
    key = C.sequence.input_variable(5, sequence_axis=seq1)
    value = C.sequence.input_variable(7, sequence_axis=seq2)

    b = ScaledDotProductAttention()(query, key, value)

    q1 = np.random.random((s3, 5)).astype(np.float32)
    q2 = np.random.random((s4, 5)).astype(np.float32)

    k1 = np.random.random((s3, 5)).astype(np.float32)
    k2 = np.random.random((s4, 5)).astype(np.float32)

    v1 = np.random.random((s1, 7)).astype(np.float32)
    v2 = np.random.random((s2, 7)).astype(np.float32)

    with pytest.raises(Exception):
        results = b.eval({query: [q1, q2], key: [k1, k2], value: [v1, v2]})


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
            # alias is used to work around bug when arguments in block funcion are the same
            attended = mha(h, encoded, C.alias(encoded))

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
    encoder_block = TransformerEncoderBlock(num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1)
    attended = encoder_block(a)

    assert attended.shape == (10, )

    n = [np.random.random((3, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32)]

    results = attended.eval({a: n})


def test_initialisation_transformer_encoder_block():
    """ custom initialise encoder block using numpy """
    model_dim = 768
    intermediate_dim = 3072
    num_heads = 12

    bias = np.random.random((model_dim, )).astype(np.float32)
    kernel = np.random.random((model_dim, model_dim)).astype(np.float32)

    intermediate_bias = np.random.random((intermediate_dim, )).astype(np.float32)
    intermediate_kernel = np.random.random((model_dim, intermediate_dim)).astype(np.float32)

    final_kernel = np.random.random((intermediate_dim, model_dim)).astype(np.float32)

    TransformerEncoderBlock(num_heads=num_heads,
                            model_dim=model_dim,
                            intermediate_dim=intermediate_dim,
                            dropout_rate=0.1,
                            obey_sequence_order=False,
                            max_seq_len=None,
                            key_init=kernel,
                            key_init_bias=bias,
                            query_init=kernel,
                            query_init_bias=bias,
                            value_init=kernel,
                            value_init_bias=bias,
                            mha_init=kernel,
                            mha_init_bias=bias,
                            mha_initial_scale=bias,
                            mha_initial_bias=bias,
                            intermediate_init=intermediate_kernel,
                            intermediate_init_bias=intermediate_bias,
                            init=final_kernel,
                            init_bias=bias,
                            initial_scale=bias,
                            initial_bias=bias)


def test_transformer_decoder_block1():
    """ default settings: encoder block output feed into decoder block """
    a = C.sequence.input_variable(10)
    b = C.sequence.input_variable(10)

    encoder_block = TransformerEncoderBlock(num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1)
    decoder_block = TransformerDecoderBlock(num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1,
                                            obey_sequence_order=False, max_seq_len=None)

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

    encoder_block = TransformerEncoderBlock(num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1)
    decoder_block = TransformerDecoderBlock(num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1,
                                            obey_sequence_order=True, max_seq_len=100)

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

    encoder_block = TransformerEncoderBlock(num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1)
    decoder_block1 = TransformerDecoderBlock(num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1,
                                             obey_sequence_order=True, max_seq_len=100)
    decoder_block2 = TransformerDecoderBlock(num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1,
                                             obey_sequence_order=True, max_seq_len=100)

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
    encoder = TransformerEncoder(n=5, num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = encoded.eval({a: n})

    encoder = TransformerEncoder(n=2, num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    results = encoded.eval({a: n})

    encoder = TransformerEncoder(n=1, num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    results = encoded.eval({a: n})


def test_transformer_encoder1c():
    """ No peeking and multi layers encoder """
    a = C.sequence.input_variable(10)
    encoder = TransformerEncoder(n=5, num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = encoded.eval({a: n})

    encoder = TransformerEncoder(n=2, num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    results = encoded.eval({a: n})

    encoder = TransformerEncoder(n=1, num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    results = encoded.eval({a: n})


def test_transformer_decoder1():
    """ default setup: 5-1 layers of decoders """
    seq1 = C.Axis.new_unique_dynamic_axis('seq1')
    seq2 = C.Axis.new_unique_dynamic_axis('seq2')

    a = C.sequence.input_variable(10, sequence_axis=seq1)
    b = C.sequence.input_variable(10, sequence_axis=seq2)

    decoder = TransformerDecoder(n=5, num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1, max_seq_len=100)

    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    m = np.random.random((4, 8, 10)).astype(np.float32)

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(n=3, num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1, max_seq_len=100)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(n=2, num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1, max_seq_len=100)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(n=1, num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1, max_seq_len=100)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})


def test_transformer_decoder2():
    """
    Different dimensions between encoded and decoder model is allowed as
    encoded will be cast to model_dim of decoder.
    """
    seq1 = C.Axis.new_unique_dynamic_axis('seq1')
    seq2 = C.Axis.new_unique_dynamic_axis('seq2')

    a = C.sequence.input_variable(30, sequence_axis=seq1)
    b = C.sequence.input_variable(10, sequence_axis=seq2)

    decoder = TransformerDecoder(n=5, num_heads=2, model_dim=10, intermediate_dim=30, dropout_rate=0.1, max_seq_len=100)

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
                              encoder_model_dim=10, decoder_model_dim=10, max_seq_len_decoder=100)

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
                              encoder_model_dim=10, decoder_model_dim=30, max_seq_len_decoder=100)

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


def test_gaussian_window_attention():
    seq1 = C.Axis.new_unique_dynamic_axis('seq1')
    seq2 = C.Axis.new_unique_dynamic_axis('seq2')

    encoded = C.sequence.input_variable(30, sequence_axis=seq1)
    query = C.sequence.input_variable(28, sequence_axis=seq2)

    a = GaussianWindowAttention(10)(encoded, query)

    n = np.random.random((2, 10, 30)).astype(np.float32)
    m = np.random.random((2, 15, 28)).astype(np.float32)

    results = a.eval({encoded: n, query: m})


def test_pretrained_bert_model1():
    """ tested to work with 'uncased_L-12_H-768_A-12' """
    text_tensor = C.sequence.input_variable(30522)
    token_type_tensor = C.sequence.input_variable(2)
    filepath_to_tf_bert_model = "../../../pretrained models/BERT/uncased/bert_model.ckpt"

    model = PreTrainedBertModel(filepath_to_tf_bert_model, 12, 0.1)
    b = model(text_tensor, token_type_tensor)

    assert b.shape == (768,)

    n1 = np.random.random((3, 30522)).astype(np.float32)
    n2 = np.random.random((6, 30522)).astype(np.float32)

    m1 = np.random.random((3, 2)).astype(np.float32)
    m2 = np.random.random((6, 2)).astype(np.float32)
    b.eval({text_tensor: [n1, n2], token_type_tensor: [m1, m2]})


def test_pretrained_bert_model2():
    """ tested to work with 'uncased_L-12_H-768_A-12' """
    text_tensor = C.sequence.input_variable(30522)
    token_type_tensor = C.sequence.input_variable(3)
    filepath_to_tf_bert_model = "../../../pretrained models/BERT/uncased/bert_model.ckpt"

    model = PreTrainedBertModel(filepath_to_tf_bert_model, 12, 0.1)

    with pytest.raises(Exception):
        model(text_tensor, token_type_tensor)
