import cntk as C
import cntkx as Cx
from cntk.layers import Recurrence
from math import pi
import numpy as np


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

    b, valid = C.sequence.unpack(x, padding_value=0).outputs
    c = C.pad(b, final_pattern, mode=mode, constant_value=constant_value)
    seq_length = C.reduce_sum(valid, axis=0) + C.Constant(sum(pattern))
    d = C.to_sequence(c, seq_length, name=name)
    return d


@C.typemap
def length(x, name=''):
    """
    Calculates the sequence length of the tensor.

    Arguments:
        x: input sequence tensor
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    """

    mask = C.sequence.unpack(x, padding_value=0).outputs[1]
    return C.reduce_sum(mask, axis=0, name=name)


def position(x, name=''):
    """ Returns the position index of every element in the sequence.

    First element of sequence will have position value of 0.

    Arguments:
        x: input sequence tensor
        name (str): name of function

    Returns:
        :class:`~cntk.ops.functions.Function`
        a sequence tensor of shape (1,) with value of 0 to `seq_length` depending on position
    """

    @C.BlockFunction('position', name)
    def inner(a):
        # reconcile_dynamic_axes is necessary to avoid subtle bugs e.g. sequence.where and one_hot
        return C.reconcile_dynamic_axes(C.sequence.where(C.ones_like(Cx.scalar(a))), a)

    return inner(x)  # {#, *] [1,]


def stride(x, s: int, tol: float = 0.1):
    """ Strides across sequential axis

    Note:
        Tested to work up to 1,000,000 sequence items. Beyond that tuning of `tol` might be necessary.

    Arguments:
        x: input sequence tensor
        s (int): sequential stride
        tol (float): tolerance due to precision error of applying `sin` function, valid seq item not exactly zero.

    Returns:
        :class:`~cntk.ops.functions.Function`
        Every `s` sequence item of `x` starting from the first sequence item

    """
    p = position(x)
    integers = p / s  # every s sequence item will be an integer
    valid = C.less_equal(C.abs(C.sin(integers * pi)), tol)  # sin of integer multiple of pi will return close to zero
    result = C.sequence.gather(x, valid)
    return result
