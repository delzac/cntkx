import cntk as C
from cntkx.layers.models import Transformer, TransformerDecoder, TransformerEncoder
import numpy as np


def test_transformer_encoder1a():
    """ Default settings: input is not seq output is seq """
    a = C.sequence.input_variable(10)
    encoder = TransformerEncoder(5, 2, 10)
    encoded = encoder(a)

    assert encoded.shape == (-3, 10)

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = encoded.eval({a: n})

    encoder = TransformerEncoder(2, 2, 10)
    encoded = encoder(a)

    assert encoded.shape == (-3, 10)

    results = encoded.eval({a: n})

    encoder = TransformerEncoder(1, 2, 10)
    encoded = encoder(a)

    assert encoded.shape == (-3, 10)

    results = encoded.eval({a: n})


def test_transformer_encoder1b():
    """ Default settings: input is not seq output is seq """
    a = C.sequence.input_variable(10)
    encoder = TransformerEncoder(5, 2, 10, output_as_seq=True)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = encoded.eval({a: n})

    encoder = TransformerEncoder(2, 2, 10, output_as_seq=True)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    results = encoded.eval({a: n})

    encoder = TransformerEncoder(1, 2, 10, output_as_seq=True)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    results = encoded.eval({a: n})


def test_transformer_encoder1c():
    """ Default settings: input is not seq output is seq """
    a = C.sequence.input_variable(10)
    encoder = TransformerEncoder(5, 2, 10, obey_sequence_order=True, max_seq_len=100, output_as_seq=True)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = encoded.eval({a: n})

    encoder = TransformerEncoder(2, 2, 10, obey_sequence_order=True, max_seq_len=100, output_as_seq=True)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    results = encoded.eval({a: n})

    encoder = TransformerEncoder(1, 2, 10, obey_sequence_order=True, max_seq_len=100, output_as_seq=True)
    encoded = encoder(a)

    assert encoded.shape == (10, )

    results = encoded.eval({a: n})


def test_transformer_decoder1a():
    """ default setup: 5 layers of decoders """
    a = C.input_variable((-3, 10))
    b = C.sequence.input_variable(10)

    decoder = TransformerDecoder(5, 2, 10, is_encoded_seq=False)

    decoded = decoder(a, b)

    assert decoded.shape == (-3, 10)

    m = np.random.random((4, 8, 10)).astype(np.float32)

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(3, 2, 10, is_encoded_seq=False)
    decoded = decoder(a, b)

    assert decoded.shape == (-3, 10)

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(2, 2, 10, is_encoded_seq=False)
    decoded = decoder(a, b)

    assert decoded.shape == (-3, 10)

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(1, 2, 10, is_encoded_seq=False)
    decoded = decoder(a, b)

    assert decoded.shape == (-3, 10)

    results = decoded.eval({a: m, b: n})


def test_transformer_decoder2():
    """ default setup: output as sequence """
    a = C.input_variable((-3, 10))
    b = C.sequence.input_variable(10)

    decoder = TransformerDecoder(5, 2, 10, is_encoded_seq=False, output_as_seq=True)

    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    m = np.random.random((4, 8, 10)).astype(np.float32)

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(3, 2, 10, is_encoded_seq=False, output_as_seq=True)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(2, 2, 10, is_encoded_seq=False, output_as_seq=True)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(1, 2, 10, is_encoded_seq=False, output_as_seq=True)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})


def test_transformer_decoder3():
    """ output as sequence with no peeking at future values """
    a = C.input_variable((-3, 10))
    b = C.sequence.input_variable(10)

    decoder = TransformerDecoder(5, 2, 10, is_encoded_seq=False, output_as_seq=True, obey_sequence_order=True, max_seq_len=100)

    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    m = np.random.random((4, 8, 10)).astype(np.float32)

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(3, 2, 10, is_encoded_seq=False, output_as_seq=True, obey_sequence_order=True, max_seq_len=100)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(2, 2, 10, is_encoded_seq=False, output_as_seq=True, obey_sequence_order=True, max_seq_len=100)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})

    decoder = TransformerDecoder(1, 2, 10, is_encoded_seq=False, output_as_seq=True, obey_sequence_order=True, max_seq_len=100)
    decoded = decoder(a, b)

    assert decoded.shape == (10, )

    results = decoded.eval({a: m, b: n})


def test_transformer():
    """ default configuration of using transformer """
    a = C.sequence.input_variable(10)
    b = C.sequence.input_variable(10)

    transformer = Transformer(num_encoder_blocks=3, num_decoder_blocks=3, num_heads_encoder=2, num_heads_decoder=2,
                              model_dim=10, encoder_obey_sequence_order=False, decoder_obey_sequence_order=True,
                              max_seq_len_encoder=100, max_seq_len_decoder=100, output_as_seq=True)

    prediction = transformer(a, b)

    n = [np.random.random((2, 10)).astype(np.float32),
         np.random.random((4, 10)).astype(np.float32),
         np.random.random((6, 10)).astype(np.float32),
         np.random.random((8, 10)).astype(np.float32)]

    results = prediction.eval({a: n, b: n})

    transformer = Transformer(num_encoder_blocks=1, num_decoder_blocks=1, num_heads_encoder=2, num_heads_decoder=2,
                              model_dim=10, encoder_obey_sequence_order=False, decoder_obey_sequence_order=True,
                              max_seq_len_encoder=100, max_seq_len_decoder=100, output_as_seq=True)

    prediction = transformer(a, b)
    results = prediction.eval({a: n, b: n})

    transformer = Transformer(num_encoder_blocks=1, num_decoder_blocks=2, num_heads_encoder=2, num_heads_decoder=2,
                              model_dim=10, encoder_obey_sequence_order=False, decoder_obey_sequence_order=True,
                              max_seq_len_encoder=100, max_seq_len_decoder=100, output_as_seq=True)

    prediction = transformer(a, b)
    results = prediction.eval({a: n, b: n})

    transformer = Transformer(num_encoder_blocks=2, num_decoder_blocks=1, num_heads_encoder=2, num_heads_decoder=2,
                              model_dim=10, encoder_obey_sequence_order=False, decoder_obey_sequence_order=True,
                              max_seq_len_encoder=100, max_seq_len_decoder=100, output_as_seq=True)

    prediction = transformer(a, b)
    results = prediction.eval({a: n, b: n})
