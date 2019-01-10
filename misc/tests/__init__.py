import numpy as np
import cntk as C
from cntkx.layers import Transformer
from cntkx.misc import greedy_decoder


def test_greedy_decoding_transformer():
    """ default configuration of using transformer """
    axis1 = C.Axis.new_unique_dynamic_axis(name='seq1')
    axis2 = C.Axis.new_unique_dynamic_axis(name='seq2')
    a = C.sequence.input_variable(10, sequence_axis=axis1)
    b = C.sequence.input_variable(10, sequence_axis=axis2)

    transformer = Transformer(num_encoder_blocks=3, num_decoder_blocks=3, num_heads_encoder=2, num_heads_decoder=2,
                              model_dim=10, encoder_obey_sequence_order=False, decoder_obey_sequence_order=True,
                              max_seq_len_encoder=100, max_seq_len_decoder=100, output_as_seq=True)

    decoded = transformer(a, b)

    input_sentence = np.random.random((7, 10)).astype(np.float32)
    start_token = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32)[None, ...]
    end_token = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)

    assert start_token.shape == (1, 10)
    assert end_token.shape == (10, )

    results = greedy_decoder(decoded, input_sentence, start_token, end_token, 100)
