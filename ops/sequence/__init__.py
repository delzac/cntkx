import cntk as C
import cntkx as Cx
from math import pi
from typing import Tuple


def pad(x, pattern: Tuple[int, int], constant_value: float = 0, name=''):
    """
    Pads a tensor in the sequence axis according to the specified patterns.
    Three padding modes are supported: CONSTANT / REFLECT / SYMMETRIC.

    Arguments:
        x: tensor to be padded.
        pattern (tuple with 2 integers): how many values to add before and after the contents in the sequence axis.
        constant_value: the value used to fill the padding cells, only meaningful under CONSTANT mode.
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    @C.BlockFunction('Sequence::Pad', name)
    def inner(a):
        paddings = [zeros_like(a, p) if p else None for p in pattern]

        if constant_value:
            paddings = [padding + constant_value if padding is not None else padding for padding in paddings]

        r = a
        if paddings[0] is not None:
            r = Cx.sequence.join(paddings[0], r)

        if paddings[1] is not None:
            r = Cx.sequence.join(r, paddings[1])
        return r

    return inner(x)


def zeros_like(x, seq_length: int):
    """ helper function to construct a sequence of zeros """
    if seq_length > 1:
        b = C.zeros_like(C.sequence.slice(x, 0, seq_length))
    elif seq_length == 1:
        b = C.to_sequence(C.expand_dims(C.zeros_like(C.sequence.first(x)), axis=C.Axis.new_leading_axis()))
    else:
        raise ValueError(f"length ({seq_length}) must be larger than 0")

    return b


@C.typemap
def length(x, name=''):
    """
    Calculates the sequence length of the tensor.

    Arguments:
        x: input sequence tensor
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
        Not a sequence tensor (i.e. no dynamic sequence axis)

    """

    @C.BlockFunction('Sequence::Length', name)
    def inner(a):
        return C.expand_dims(C.sequence.reduce_sum(C.sequence.broadcast_as(1, a)), axis=C.Axis.new_leading_axis())

    return inner(x)  # shape: [#] [1, ]


def position(x, name=''):
    """ Returns the position index of every element in the sequence.

    First element of sequence will have position value of 0.

    Example:
        a = C.sequence.input_variable(10)
        b = Cx.sequence.position(a)

        assert b.shape == (1,)

    Arguments:
        x: input sequence tensor
        name (str): name of function

    Returns:
        :class:`~cntk.ops.functions.Function`
        a sequence tensor of shape (1,) with value from 0 (first seq item) to `seq_length` - 1 (last seq item)
    """

    @C.BlockFunction('Sequence::Position', name)
    def inner(a):
        # reconcile_dynamic_axes is necessary to avoid subtle bugs e.g. sequence.where and one_hot
        return C.expand_dims(C.reconcile_dynamic_axes(C.sequence.where(C.sequence.broadcast_as(1, a)), a), axis=-1)

    return inner(x)  # {#, *] [1,]


def stride(x, s, name=''):
    """ Strides across sequential axis, picking up every s element start from the first sequential element.

    Example:
        seq: [0, 1, 2, 3, 4, 5]
        after stride(seq, 2): [0, 2, 4]

    Arguments:
        x: input sequence tensor
        s (int): sequential stride
        name (str): name of function

    Returns:
        :class:`~cntk.ops.functions.Function`
        Every `s` sequence item of `x` starting from the first sequence item

    """
    @C.BlockFunction('Sequence::Stride', name)
    def inner(a):
        p = position(a)
        quotient = p / s  # every s sequence item will be an integer
        decimals = quotient - C.floor(quotient)  # every s sequence item will be a zero
        valid = C.equal(decimals, 0)
        result = C.sequence.gather(a, valid)
        return result

    return inner(x)


def join(x, y, name=''):
    """ joins two sequences along their dynamic sequence axis. Static axis between a and b
    must be the same and the dimensions of the static axes will remain unchanged in the op.

    Example:
        import cntk as C
        import cntkx as Cx

        a = C.sequence.input_variable(3)
        b = C.sequence.input_variable(3)

        ab = Cx.sequence.join(a, b)

        assert ab.shape == a.shape == b.shape == (3, )

    Arguments:
        x: Sequence tensor
        y: Sequence tensor
        name (str): name of function

    Returns:
        :class:`~cntk.ops.functions.Function`
        A new sequence tensor with sequence axis that is the concatenation of the seq axis of a and b

    """

    @C.BlockFunction("Sequence::Join", name)
    def inner(a, b):
        a_unpacked, a_mask = C.sequence.unpack(a, padding_value=0).outputs
        b_unpacked, b_mask = C.sequence.unpack(b, padding_value=0).outputs

        ab_unpacked = C.splice(a_unpacked, b_unpacked, axis=0)
        ab_mask = C.expand_dims(C.splice(a_mask, b_mask), axis=-1)

        ab_w_pad = C.to_sequence(ab_unpacked)
        ab_condition = C.to_sequence(ab_mask)

        ab = C.sequence.gather(ab_w_pad, ab_condition)
        return ab

    return inner(x, y)


def window(x, width: int, slide: int, new_axis=False, name=''):
    """ Creates a non-causal window in the sequence tensor. Window contains future values.

    It effectively reduces the sequence length by `slide` factor while increasing tensor dimension by `width` factor.
    Useful to reduce computation workload in recurrent networks. Used in pyramidal BLSTM in acoustic modelling.

    Graphic:
        sequence: [0, 1, 2, 3, 4, 5, 6, 7]
        window(sequence, width=2, slide=2)

        output: [[0, 2, 4, 6]
                 [1, 3, 5, 7]]


    Example:
        width = 2
        slide = 2
        a = C.sequence.input_variable(10)
        b = Cx.sequence.window(a, width, slide)

        assert b.shape == (10 * k, )  # while sequence length reduces by a factor of `slide`

    Arguments:
        x: input tensor
        width: width of window
        slide: striding length along the sequential axis
        new_axis (bool): whether to concatenate to a new static axis or concatenate to the last axis
        name (str): name of function

    Returns:
        :class:`~cntk.ops.functions.Function`
        A new sequence tensor with sequence length by `slide` factor with tensor dimension increased by `width` factor
    """

    @C.BlockFunction('Sequence::Window', name)
    def inner(a):
        future = [C.sequence.future_value(a, time_step=1 + i) for i in range(width - 1)]
        frames = C.splice(a, *future, axis=C.Axis.new_leading_axis() if new_axis else -1)
        y = stride(frames, slide) if slide > 1 else frames
        return y

    return inner(x)


def window_causal(x, width: int, slide: int, new_axis=False, name=''):
    """ Creates a non-causal window in the sequence tensor. Window contains future values.

    It effectively reduces the sequence length by `slide` factor while increasing tensor dimension by `width` factor.
    Useful to reduce computation workload in recurrent networks, or to convolution across sequence axis.

    Note:
        When using `window_causal`, there's a possibility that the last few sequence item might get leftout,
        compared to using `window` above.

    Graphic:
        sequence: [0, 1, 2, 3, 4, 5, 6, 7]
        window(sequence, width=2, slide=2)

        output: [[0, 2, 4, 6]
                 [0, 1, 3, 5]]

        sequence item 7 gets left out

    Example:
        width = 2
        slide = 2
        a = C.sequence.input_variable(10)
        b = Cx.sequence.window_causal(a, width, slide)

        assert b.shape == (10 * k, )  # while sequence length reduces by a factor of `slide`

    Arguments:
        x: input tensor
        width: width of window
        slide: striding length along the sequential axis
        new_axis (bool): whether to concatenate to a new static axis or concatenate to the last axis
        name (str): name of function

    Returns:
        :class:`~cntk.ops.functions.Function`
        A new sequence tensor with sequence length by `slide` factor with tensor dimension increased by `width` factor
    """
    @C.BlockFunction('Sequence::SlidingWindow', name)
    def inner(a):
        history = list(reversed([C.sequence.past_value(a, time_step=i + 1) for i in range(width - 1)]))
        frames = C.splice(*history, a, axis=C.Axis.new_leading_axis() if new_axis else -1)
        y = stride(frames, slide) if slide > 1 else frames
        return y

    return inner(x)


def reverse(x, name=''):
    """ Reverses the items in sequence axis

    This function is used to build a Bidirectional Auto-regressive rnn layer. Using UnfoldFrom with
    Recurrence(x) and Recurrence(x, go_backwards=True) will result in 'ValueError: It is not allowed to
    have multiple different stepping directions in the same loop'.

    To workaround, instead of reversing in Recurrence(), we reverse the input sequence instead.

    Example:
        import cntk as C
        import cntkx as Cx
        from cntk.layers import Recurrence, UnfoldFrom, LSTM

        hidden_dim = 50
        start_token = C.Constant(0, shape=(hidden_dim,))
        a = C.sequence.input_variable(1, name='seq1')

        b = UnfoldFrom(Recurrence(LSTM(hidden_dim), go_backwards=True))(start_token, a)

        n = [np.random.random((10, hidden_dim)).astype(np.float32),]

        # This raise 'ValueError: It is not allowed to have multiple different stepping directions in the same loop'
        b.eval({b.arguments[0]: n})


    Example:
        import cntk as C
        import cntkx as Cx
        from cntk.layers import Recurrence, UnfoldFrom, LSTM

        hidden_dim = 50
        start_token = C.Constant(0, shape=(hidden_dim,))
        a = C.sequence.input_variable(1, name='seq1')
        a_reversed = Cx.sequence.reverse(a)

        b = UnfoldFrom(Recurrence(LSTM(hidden_dim)))(start_token, a_reversed)  # remove go_backwards=True

        n = [np.random.random((10, hidden_dim)).astype(np.float32),]
        b.eval({b.arguments[0]: n})  # this will run just fine

    Arguments:
        x: input tensor
        name (str): name of function

    Returns:
        :class:`~cntk.ops.functions.Function`
        `x` with its sequence axis reversed

    """

    @C.BlockFunction('Sequence::Reverse', name)
    def inner(a):
        values, valid = C.sequence.unpack(a, padding_value=0).outputs
        values_reversed = C.slice(values, 0, 0, 0, -1)
        valid_reversed = C.slice(valid, 0, 0, 0, -1)

        values_seq = C.to_sequence(values_reversed)
        valid_seq = C.to_sequence(C.expand_dims(valid_reversed, axis=-1))
        a_reversed = C.sequence.gather(values_seq, valid_seq)
        return a_reversed

    return inner(x)


def reduce_mean(seq, name=''):
    """ Computes the mean of the input sequence's elements across the sequence axis.

    Examples:
        import cntk as C
        import cntkx as Cx

        a = C.sequence.input_variable((3, 4))
        b = Cx.sequence.reduce_mean(a)

        n = [np.random.random((10, 3, 4)).astype(np.float32),]
        results = b.eval({a: n})

        for r, d in zip(results, n):
            np.testing.assert_almost_equal(r, np.mean(d, axis=0))


    Args:
        seq: sequence input tensor
        name (`str`, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    """

    @C.BlockFunction('Sequence::ReduceMean', name)
    def inner(a):
        b = C.sequence.reduce_sum(a)
        c = b / Cx.sequence.length(a)
        return c

    return inner(seq)


def reduce_concat_pool(x, axis=0, name=''):
    """ Reduce concat pooling: concatenates the last seq item with the reduce_max and reduce_mean of the sequence axis.
    This is can be used as a drop-in replacement anytime sequence.last is used. It will provide superior performance
    compared to it.

    Examples:
        n = 32
        a = C.sequence.input_variable(n)
        b = Cx.sequence.reduce_concat_pool(a)

        assert b.shape == (n * 3, )

    Arguments:
        x: input tensor
        axis: concatenation axis
        name (`str`, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    @C.BlockFunction('Sequence::ReduceConcatPool', name)
    def inner(a):
        return C.splice(C.sequence.last(a), C.sequence.reduce_max(a), Cx.sequence.reduce_mean(a), axis=axis)

    return inner(x)
