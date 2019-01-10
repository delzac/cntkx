import cntk as C
import numpy as np


##########################################################################
# wrapper
##########################################################################
def greedy_decoder(decoder, input_sequence, start_token, end_token):
    """ Greedy decoder wrapper for Transformer decoder. Pure python loop. One batch (sample) at a time.

    Arguments:
        decoder: :class:`~cntk.ops.functions.Function`
        input_sequence: one hot encoded 2d numpy array
        start_token: one hot encoded numpy array 2d
        end_token: one hot encoded numpy array 2d

    Returns:
        list of 2d numpy array
    """

    assert isinstance(input_sequence, np.ndarray)
    assert isinstance(start_token, np.ndarray)
    assert isinstance(end_token, np.ndarray)

    assert end_token.ndim == 1

    @C.Function
    def hardmax(x):
        return C.round(C.softmax(x, axis=-1))

    if len(decoder.shape) == 1:
        greedy_decoder = decoder >> C.hardmax
    else:
        greedy_decoder = decoder >> hardmax

    zero = np.zeros_like(input_sequence, dtype=np.float32)[:-1, ...]
    dummy_decode_seq = [start_token]

    a = [input_sequence]
    for i in range(input_sequence.shape[0]):
        results = greedy_decoder.eval({greedy_decoder.arguments[0]: a, greedy_decoder.arguments[1]: dummy_decode_seq})
        dummy_decode_seq[0] = np.concatenate((dummy_decode_seq[0], results[0][i][None, ...]), axis=0)
        # print(dummy_decode_seq[0])

        if np.all(results[0][i, ...] == end_token):
            print("completed")
            break

    return dummy_decode_seq
