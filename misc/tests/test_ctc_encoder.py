from .. import CTCEncoder
import cntk as C
from cntk.layers import Dense, LSTM, Recurrence
import numpy as np


def test_ctc_encoder_string_labels():
    label_set = list('abc')
    encoder = CTCEncoder(label_set)

    seq_length = 7
    labels = list('abbbc')
    actual = encoder.transform(labels, seq_length)

    desired = [[0, 2, 0, 0],
               [0, 0, 2, 0],
               [0, 0, 1, 0],
               [0, 0, 2, 0],
               [0, 0, 1, 0],
               [0, 0, 2, 0],
               [0, 0, 0, 2],]

    desired = np.array(desired)

    np.testing.assert_equal(actual, desired)

    # test inverse
    decoded = encoder.inverse_transform(actual)
    assert labels == decoded

    seq_length = 10
    labels = list('abbbc')
    actual = encoder.transform(labels, seq_length)

    desired = [[0, 2, 0, 0],
               [0, 0, 2, 0],
               [0, 0, 1, 0],
               [0, 0, 2, 0],
               [0, 0, 1, 0],
               [0, 0, 2, 0],
               [0, 0, 0, 2],
               [0, 0, 0, 1],
               [0, 0, 0, 1],
               [0, 0, 0, 1],]

    desired = np.array(desired)

    np.testing.assert_equal(actual, desired)

    # test inverse
    decoded = encoder.inverse_transform(actual)
    assert labels == decoded

    seq_length = 10
    labels = list('abbba')
    actual = encoder.transform(labels, seq_length)

    desired = [[0, 2, 0, 0],
               [0, 0, 2, 0],
               [0, 0, 1, 0],
               [0, 0, 2, 0],
               [0, 0, 1, 0],
               [0, 0, 2, 0],
               [0, 2, 0, 0],
               [0, 1, 0, 0],
               [0, 1, 0, 0],
               [0, 1, 0, 0], ]

    desired = np.array(desired)

    np.testing.assert_equal(actual, desired)

    # test inverse
    decoded = encoder.inverse_transform(actual)
    assert labels == decoded


def test_ctc_encoder_int_labels():
    label_set = list(range(3))
    encoder = CTCEncoder(label_set)

    seq_length = 7
    labels = [1, 2, 2, 2, 3]
    actual = encoder.transform(labels, seq_length)

    desired = [[0, 2, 0, 0],
               [0, 0, 2, 0],
               [0, 0, 1, 0],
               [0, 0, 2, 0],
               [0, 0, 1, 0],
               [0, 0, 2, 0],
               [0, 0, 0, 2],]

    desired = np.array(desired)

    np.testing.assert_equal(actual, desired)

    # test inverse
    decoded = encoder.inverse_transform(actual)
    assert labels == decoded

    seq_length = 10
    labels = [1, 2, 2, 2, 3]
    actual = encoder.transform(labels, seq_length)

    desired = [[0, 2, 0, 0],
               [0, 0, 2, 0],
               [0, 0, 1, 0],
               [0, 0, 2, 0],
               [0, 0, 1, 0],
               [0, 0, 2, 0],
               [0, 0, 0, 2],
               [0, 0, 0, 1],
               [0, 0, 0, 1],
               [0, 0, 0, 1],]

    desired = np.array(desired)

    np.testing.assert_equal(actual, desired)

    # test inverse
    decoded = encoder.inverse_transform(actual)
    assert labels == decoded

    seq_length = 10
    labels = [1, 2, 2, 2, 1]
    actual = encoder.transform(labels, seq_length)

    desired = [[0, 2, 0, 0],
               [0, 0, 2, 0],
               [0, 0, 1, 0],
               [0, 0, 2, 0],
               [0, 0, 1, 0],
               [0, 0, 2, 0],
               [0, 2, 0, 0],
               [0, 1, 0, 0],
               [0, 1, 0, 0],
               [0, 1, 0, 0], ]

    desired = np.array(desired)

    np.testing.assert_equal(actual, desired)

    # test inverse
    decoded = encoder.inverse_transform(actual)
    assert labels == decoded


def test_ctc_encoder_train_and_network_output_to_labels():
    # test CTC encoder in training loop and CTCEncoder.network_output_to_labels

    a = C.sequence.input_variable(10)
    labels = ['a', 'b', 'c']
    encoder = CTCEncoder(labels)

    labels_tensor = C.sequence.input_variable(len(encoder.classes_))  # number of classes = 4
    input_tensor = C.sequence.input_variable(100)

    prediction_tensor = Dense(4)(Recurrence(LSTM(100))(C.ones_like(input_tensor)))

    labels_graph = C.labels_to_graph(labels_tensor)

    fb = C.forward_backward(labels_graph, prediction_tensor, blankTokenId=encoder.blankTokenId)

    ground_truth = ['a', 'b', 'b', 'b', 'c']
    seq_length = 10  # must be the same length as the sequence length in network_out

    pred = np.array([[0., 2., 0., 0.],
                     [0., 2., 0., 0.],
                     [0., 0., 2., 0.],
                     [2., 0., 0., 0.],
                     [0., 0., 2., 0.],
                     [2., 0., 0., 0.],
                     [0., 0., 2., 0.],
                     [2., 0., 0., 0.],
                     [0., 0., 0., 2.],
                     [0., 0., 0., 2.], ]).astype(np.float32)

    n = np.random.random((10, 100)).astype(np.float32)

    # result = fb.eval({labels_tensor: [encoder.transform(ground_truth, seq_length=seq_length)],
    #                   input_tensor: [n]})

    # print(result)

    adam = C.adam(prediction_tensor.parameters, 0.01, 0.912)
    trainer = C.Trainer(prediction_tensor, (fb,), [adam])

    for i in range(300):
        trainer.train_minibatch({labels_tensor: [encoder.transform(ground_truth, seq_length=seq_length)],
                                 input_tensor: [n]})

        # print(trainer.previous_minibatch_loss_average)

    result = prediction_tensor.eval({input_tensor: [n]})
    assert encoder.network_output_to_labels(result[0], squash_repeat=True) == ground_truth
