import cntk as C
from cntk.layers import Recurrence, Dense


def pad(x, pattern, mode=C.CONSTANT_PAD, constant_value=0, name=''):
    """
    Pads a tensor in the sequence axis according to the specified patterns.
    Three padding modes are supported: CONSTANT / REFLECT / SYMMETRIC.

    Arguments:
        x: tensor to be padded.
        pattern (tuple with 2 integers): how many values to add before and after the contents in the sequence axis.
        mode (int): padding mode: C.ops.CONSTANT_PAD, C.ops.REFLECT_PAD and C.ops.SYMMETRIC_PAD
        constant_value: the value used to fill the padding cells, only meaningful under CONSTANT mode.
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    if not all(isinstance(i, int) for i in pattern) or not isinstance(pattern, tuple):
        raise ValueError(f"pattern {pattern} must be a tuple with 2 integers")

    ndim = len(x.shape)
    null_pattern = [(0, 0)] * ndim
    b = C.sequence.unpack(x, padding_value=0, no_mask_output=True)
    c = C.pad(b, [pattern] + null_pattern, mode=mode, constant_value=constant_value, name=name)
    d = C.to_sequence(c, length(x) + C.Constant(sum(pattern)))
    return d


@C.Function
def length(x):
    """
    Calculates the sequence length of the tensor.

    Arguments:
         x: input sequence tensor
    """

    def step(acc, a):
        return 1 + acc + a * 0

    return C.sequence.last(Recurrence(step)(x))
