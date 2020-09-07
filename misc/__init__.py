import numpy as np
from typing import List, Union
from sklearn.preprocessing import LabelBinarizer
from os.path import join, basename


def to_ctc_encoded(labels: np.ndarray) -> np.ndarray:
    """ Convert standard one hot encoded labels into ctc encoded that is compatible with ctc training in cntk.
    All the 1s in labels will be replace with 2s. And any consecutive repeated labels will have a fake label
    inserted between then with the value 1.

    Arguments:
        labels (np.ndarray): numpy array labels that is already one hot encoded

    Returns:
        float32 ctc encoded labels that would be compatible with ctc training in cntk

    """

    # convert 1s to 2s. 2 denoted frame boundary
    labels[labels == 1] = 2

    # insert fake second frame if there are repeated labels adjacent to each other
    double = [(i, a) for i, (a, b) in enumerate(zip(labels[:-1], labels[1:])) if np.all(a == b)]

    if len(double) > 0:
        indices, values = zip(*double)
        values = [value / 2 for value in values]  # 1 to indicate within phone boundary
        indices = [i + 1 for i in indices]  # np inserts before index
        labels = np.insert(labels, indices, values, axis=0)

    return labels


class CTCEncoder:
    """ Class to help convert data into an acceptable format for ctc training.

    CNTK's CTC implementation requires that data be formatted in a particular way that's typically in acoustic
    modeling but unusual in other applications. So class provides an easy way to convert data between
    what users typically expect and what cntk demands.

    Example:
        labels = ['a', 'b', 'c']
        encoder = CTCEncoder(labels)

        labels_tensor = C.sequence.input_variable(len(encoder.classes_))  # number of classes = 4
        input_tensor = C.sequence.input_variable(100)

        labels_graph = C.labels_to_graph(labels_tensor)
        network_out = model(input_tensor)

        fb = C.forward_backward(labels_graph, network_out, blankTokenId=encoder.blankTokenId)

        ground_truth = ['a', 'b', 'b', 'b', 'c']
        seq_length = 10  # must be the same length as the sequence length in network_out

        fb.eval({input_tensor: [...],
                 labels_tensor: [encoder.transform(ground_truth, seq_length=seq_length)]})

    """

    def __init__(self, labels: List[Union[str, int]]):
        """

        Arguments:
            labels (List[Union[str, int]]): labels can either be a list of ints representing the class index or
              a list of str representing the name of the class directly

        """
        self.ctc_blank = '<CTC_BLANK>' if all(isinstance(l, str) for l in labels) else max(labels) + 1

        self.label_binarizer = LabelBinarizer(pos_label=2)
        self.label_binarizer.fit(labels + [self.ctc_blank])
        self.classes_ = self.label_binarizer.classes_
        self.blankTokenId = self.classes_.tolist().index(self.ctc_blank)

    def transform(self, labels: List[Union[str, int]], seq_length: int) -> np.ndarray:
        """ Transform labels into ground truth data acceptable by cntk's forward-backward

        Arguments:
            labels (List[Union[str, int]]): list of string or int representing the labels/class
            seq_length (int): length of sequence to be padded until (seq length must be same as seq length in model output)

        Returns:
            np.ndarray
            Padded sequence array that is ready to be consume by cntk's forward-backward

        """
        labels_binarized = self.label_binarizer.transform(labels)
        labels_binarized = to_ctc_encoded(labels_binarized)

        if seq_length < labels_binarized.shape[0]:
            raise ValueError(f"seq_length ({seq_length}) is shorter than ctc labels ({labels_binarized.shape[0]}). It must be equal or larger after frame padding.")

        # pad to sequence length
        sequence = np.zeros(shape=(seq_length, labels_binarized.shape[1]))
        sequence[:labels_binarized.shape[0], ...] = labels_binarized
        sequence[labels_binarized.shape[0]:, labels_binarized[-1].argmax()] = 1
        return sequence.astype(np.float32)

    def inverse_transform(self, encoded: np.ndarray) -> List[Union[str, int]]:
        """ Inverse operation of transform

        Arguments:
            encoded (np.ndarray): numpy 2d array

        Returns:
            List[Union[str, int]]
            The labels that would result in encoded if labels feed into transform()
        """
        mask = np.sum(encoded, axis=1) != 1
        labels = encoded[mask, ...]
        labels = self.label_binarizer.inverse_transform(labels)
        return labels.tolist()

    def network_output_to_labels(self, network_output: np.ndarray, squash_repeat=True) -> List[Union[str, int]]:
        """ Parse model network output into labels that are human readable

        Network output after ctc training is not in the same format as what transform produces.

        Arguments:
            network_output (np.ndarray): outputs from network model (output layer should have no activation)
            squash_repeat (bool): whether to merge sequences of identical samples. If true then "-aa--abb" will be
                                  squash to "-a-ab"

        Returns:
            Labels (list of label)

        """
        assert network_output.ndim == 2, "expect shape (seq_length, classes + blank)"

        labels = self.label_binarizer.inverse_transform(network_output).tolist()

        if squash_repeat:
            labels.append('END99999')  # append dummy at end to preserve length of list
            labels = [i for i, j in zip(labels[:-1], labels[1:]) if i != j]

        labels = [l for l in labels if l != self.ctc_blank]
        return labels


def convert_pytorch_state_dict_to_h5_file(model_file_path: str, save_directory: str):
    try:
        import torch
        import h5py
    except ImportError:
        raise ImportError(f'Please install Pytorch and h5py first to use this function')

    h5_file_path = join(save_directory, f'{basename(model_file_path)}.hdf5')

    data = torch.load(model_file_path, map_location=lambda storage, location: storage)
    h5f = h5py.File(h5_file_path, 'w')

    for key, value in data.items():
        h5f.create_dataset(key, data=value.numpy())

    h5f.close()
    return None


##########################################################################
# wrapper
##########################################################################
def greedy_decoder(decoder, input_sequence, start_token, end_token, max_seq_len: int):
    import cntk as C
    import cntkx as Cx

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
