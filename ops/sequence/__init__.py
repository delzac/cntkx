import cntk as C
from cntk.internal import sanitize_input
import cntkx as Cx
from math import pi


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

    @C.BlockFunction('Sequence::Position', name)
    def inner(a):
        # reconcile_dynamic_axes is necessary to avoid subtle bugs e.g. sequence.where and one_hot
        return C.reconcile_dynamic_axes(C.sequence.where(C.ones_like(Cx.scalar(a))), a)

    return inner(x)  # {#, *] [1,]


def stride(x, s, tol=0.2, name=''):
    """ Strides across sequential axis

    Note:
        Tested to work up to 1,000,000 sequence items. Beyond that tuning of `tol` might be necessary.

    Arguments:
        x: input sequence tensor
        s (int): sequential stride
        tol (float): tolerance due to precision error of applying `sin` function, valid seq item not exactly zero.
        name (str): name of function

    Returns:
        :class:`~cntk.ops.functions.Function`
        Every `s` sequence item of `x` starting from the first sequence item

    """
    @C.BlockFunction('Sequence::Stride', name)
    def inner(a):
        p = position(a)
        integers = p / s  # every s sequence item will be an integer
        valid = C.less_equal(C.abs(C.sin(integers * pi)), tol)  # sin of integer multiple of pi will return close to zero
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


def window(x, width, new_axis=False, name=''):
    """ Creates a non-overlapping window in sequence tensor

    It effectively reduces the sequence length by k factor while increasing tensor dimension by k factor.
    Useful to reduce computation workload in recurrent networks. Used in pyramidal BLSTM in acoustic modelling.

    Example:
        width = 2
        a = C.sequence.input_variable(10)
        b = Cx.sequence.window(a, width)

        assert b.shape == (10 * k, )  # while sequence length reduces by a factor of k

    Arguments:
        x: input tensor
        width: width of window
        new_axis (bool): whether to concatenate to a new static axis or concatenate to the last axis
        name (str): name of function

    Returns:
        :class:`~cntk.ops.functions.Function`
        A new sequence tensor with sequence length by k factor with tensor dimension increased by k factor

    """

    @C.BlockFunction('Sequence::Window', name)
    def inner(a):
        w = [a] + [C.sequence.future_value(a, time_step=1 + i) for i in range(width - 1)]
        w = C.splice(*w, axis=C.Axis.new_leading_axis() if new_axis else -1)
        y = stride(w, width)
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
