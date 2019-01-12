import cntk as C
import cntkx as Cx
import numpy as np


##########################################################################
# wrapper
##########################################################################
def greedy_decoder(decoder, input_sequence, start_token, end_token, max_seq_len: int):
    """ Greedy decoder wrapper for Transformer decoder. Pure python loop. One batch (sample) at a time.

    Example:
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

    Arguments:
        decoder: :class:`~cntk.ops.functions.Function`
        input_sequence: one hot encoded 2d numpy array
        start_token: one hot encoded numpy array 2d
        end_token: one hot encoded numpy array 2d
        max_seq_len: max sequence length to run for without encountering end token
    Returns:
        list of 2d numpy array
    """

    assert isinstance(input_sequence, np.ndarray)
    assert isinstance(start_token, np.ndarray)
    assert isinstance(end_token, np.ndarray)

    assert end_token.ndim == 1

    if len(decoder.shape) == 1:
        greedy_decoder = decoder >> C.hardmax
    else:
        greedy_decoder = decoder >> Cx.hardmax  # hardmax applied on axis=-1

    dummy_decode_seq = [start_token]

    a = [input_sequence]
    for i in range(max_seq_len):
        results = greedy_decoder.eval({greedy_decoder.arguments[0]: a, greedy_decoder.arguments[1]: dummy_decode_seq})
        dummy_decode_seq[0] = np.concatenate((dummy_decode_seq[0], results[0][i][None, ...]), axis=0)
        # print(dummy_decode_seq[0])

        if np.all(results[0][i, ...] == end_token):
            print("completed")
            break

    return dummy_decode_seq
