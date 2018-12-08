import cntk as C
from cntk.layers import Recurrence


@C.typemap
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
    final_pattern = [pattern] + null_pattern

    b = C.sequence.unpack(x, padding_value=0, no_mask_output=True)
    c = C.pad(b, final_pattern, mode=mode, constant_value=constant_value)
    seq_length = length(x) + C.Constant(sum(pattern))
    d = C.to_sequence(c, seq_length, name=name)
    return d


@C.typemap
def length(x, name=''):
    """
    Calculates the sequence length of the tensor.

    Arguments:
         x: input sequence tensor
         name (str, optional): the name of the Function instance in the network
    """

    def step(acc, a):
        return 1 + acc + a * 0

    return C.sequence.last(Recurrence(step)(C.slice(x, 0, 0, 1, name=name)))
