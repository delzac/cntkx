from .. import CTCEncoder
import numpy as np


def test_ctc_encoder_transform():
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


def test_ctc_encoder_transform1():
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
