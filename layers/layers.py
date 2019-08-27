import math
import numpy as np
import cntk as C
import cntkx as Cx
from cntkx.layers.blocks import _INFERRED
from cntk.default_options import default_override_or, get_default_override
from cntk.layers.blocks import identity, _initializer_for
from cntk.layers import Dropout
from cntk.layers import MaxPooling, LayerNormalization, AveragePooling
from cntk.internal import _as_tuple
from cntk.variables import Record


def _window(x, axis, begin, end, step, stride, initial_state=None):
    '''
    helper to expand a sequence into a window, splicing them along the given axis (which must already exist)
    '''
    shifted = [
        C.sequence.past_value(x, initial_state=initial_state, time_step=-t) if t < 0 else
        x                                                        if t == 0 else
        C.sequence.future_value(x, initial_state=initial_state, time_step=t)
        for t in range(begin, end, step)
    ]
    r = C.splice(*shifted, axis=axis)
    if stride != 1:
        raise NotImplementedError('windowed convolution with stride not yet implemented')
    return r


# helper to expand options that can be specified as a single value
def _pad_to_shape(filter_shape, param, what):
    param = _as_tuple(param)
    if len(param) == 1: # broadcast
        while len(param) < len(filter_shape):
            param = (param[0],) + param
    if len(param) != len(filter_shape):
        raise ValueError("{} parameter ({}) must be a scalar or have same number of elements as the filter_shape parameter ({})".format(what, param, filter_shape))
    return param


# TODO: To be removed once the main cntk line accepts the pull request to fix initialisation bug
def Dense(shape, activation=default_override_or(identity), init=default_override_or(C.glorot_uniform()),
          input_rank=None, map_rank=None, bias=default_override_or(True), init_bias=default_override_or(0), name=''):
    '''
    Dense(shape, activation=identity, init=glorot_uniform(), input_rank=None, map_rank=None, bias=True, init_bias=0, name='')

    Layer factory function to create an instance of a fully-connected linear layer of the form
    `activation(input @ W + b)` with weights `W` and bias `b`, and `activation` and `b` being optional.
    `shape` may describe a tensor as well.

    A ``Dense`` layer instance owns its parameter tensors `W` and `b`, and exposes them as attributes ``.W`` and ``.b``.

    Example:
     >>> f = Dense(5, activation=C.relu)
     >>> x = C.input_variable(3)
     >>> h = f(x)
     >>> h.shape
         (5,)
     >>> f.W.shape
         (3, 5)
     >>> f.b.value
         array([ 0.,  0.,  0.,  0.,  0.], dtype=float32)

     >>> # activation through default options
     >>> with C.default_options(activation=C.relu):
     ...     f = Dense(500)

    The ``Dense`` layer can be applied to inputs that are tensors, not just vectors.
    This is useful, e.g., at the top of a image-processing cascade, where after many
    convolutions with padding and strides it is difficult to know the precise dimensions.
    For this case, CNTK has an extended definition of matrix product, in which
    the input tensor will be treated as if it had been automatically flattened.
    The weight matrix will be a tensor that reflects the "flattened" dimensions in its axes.

    Example:
     >>> f = Dense(5, activation=C.softmax) # a 5-class classifier
     >>> x = C.input_variable((64,16,16)) # e.g. an image reduced by a convolution stack
     >>> y = f(x)
     >>> y.shape
     (5,)
     >>> f.W.shape  # "row" dimension of "matrix" consists of 3 axes that match the input
     (64, 16, 16, 5)

    This behavior can be modified by telling CNTK either the number of axes that should not be projected (``map_rank``)
    or the rank of the input (``input_rank``). If neither is specified, all input dimensions are
    projected, as in the example above.

    Example:
     >>> f = Dense(5, activation=C.softmax, input_rank=2) # a 5-class classifier
     >>> x = C.input_variable((10, 3, 3)) # e.g. 10 parallel 3x3 objects. Input has input_rank=2 axes
     >>> y = f(x)
     >>> y.shape  # the 10 parallel objects are classified separately, the "10" dimension is retained
     (10, 5)
     >>> f.W.shape  # "row" dimension of "matrix" consists of (3,3) matching the input axes to project
     (3, 3, 5)

     >>> f = Dense(5, activation=C.softmax, map_rank=2)
     >>> x = C.input_variable((4, 6, 3, 3, 3)) # e.g. 24 parallel 3x3x3 objects arranged in a 4x6 grid. The grid is to be retained
     >>> y = f(x)
     >>> y.shape  # the 4x6 elements are classified separately, the grid structure is retained
     (4, 6, 5)
     >>> f.W.shape  # "row" dimension of "matrix" consists of (3,3) matching the input axes to project
     (3, 3, 3, 5)
     >>> z = y([np.zeros(x.shape)])
     >>> assert z.shape == (1, 4, 6, 5)

    Args:
     shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
     activation (:class:`~cntk.ops.functions.Function`, defaults to identity): optional function to apply at the end, e.g. `relu`
     init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
     input_rank (int, defaults to `None`): number of inferred axes to add to W (`map_rank` must not be given)
     map_rank (int, defaults to `None`): expand W to leave exactly `map_rank` axes (`input_rank` must not be given)
     bias (bool, optional, defaults to `True`): the layer will have no bias if `False` is passed here
     init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the operation to it
    '''

    activation = C.get_default_override(Dense, activation=activation)
    init       = C.get_default_override(Dense, init=init)
    bias       = C.get_default_override(Dense, bias=bias)
    init_bias  = C.get_default_override(Dense, init_bias=init_bias)

    output_shape = _as_tuple(shape)

    if input_rank is not None and map_rank is not None:
        raise ValueError("Dense: input_rank and map_rank cannot be specified at the same time.")

    # determine meaning of axes
    # W gets dimension (input_shape + shape)
    # where input_shape is determined as:
    #  - by default, equal to the dimensions of the input passed to Dense()
    #  - if input_rank is given, then the last 'input_rank' dimensions of the input (all others are not reduced over)
    #  - if map_rank is given, then the all but the first 'map_rank' dimensions of the input (those are not reduced over)
    # where input_rank and map_rank are mutually exclusive.

    output_rank = len(output_shape)   # support outputs with tensor layouts

    # If input_rank not given then pass a single _INFERRED; map_rank if given will determine the input_rank.
    # The dimension inference may still create multiple axes.
    input_shape = _INFERRED * (input_rank if input_rank is not None else 1)

    if input_rank is not None:
        infer_input_rank_to_map = -1 # means map_rank is not specified; input_rank rules
    elif map_rank is None:
        infer_input_rank_to_map = 0  # neither given: default to 'infer W to use all input dims'
    else:
        infer_input_rank_to_map = map_rank  # infer W to use all input dims except the first static 'map_rank' ones

    # parameters bound to this Function
    if isinstance(init, np.ndarray):
        init_weights = init
    else:
        init_weights = _initializer_for(init, Record(output_rank=output_rank))

    W = C.Parameter(input_shape + output_shape, init=init_weights, name='W')
    b = C.Parameter(              output_shape, init=init_bias,    name='b') if bias else None

    # expression of this function
    @C.BlockFunction('Dense', name)
    def dense(x):
        r = C.times(x, W, output_rank=output_rank, infer_input_rank_to_map=infer_input_rank_to_map)
        if b:
            r = r + b
        if activation is not None:
            r = activation(r)
        return r
    return dense


def Embedding(shape=None, init=default_override_or(C.glorot_uniform()), weights=None, enable_weight_tying=False, name=''):
    '''
    Embedding(shape=None, init=glorot_uniform(), weights=None, enable_weight_tying=False, name='')

    Layer factory function to create a embedding layer.

    An embedding is conceptually a lookup table. For every input token (e.g. a word or any category label), the corresponding
    entry in the lookup table is returned.

    In CNTK, discrete items such as words are represented as one-hot vectors.
    The table lookup is realized as a matrix product, with a matrix
    whose rows are the embedding vectors.
    Note that multiplying a matrix from the left with a one-hot vector is the same as copying
    out the row for which the input vector is 1.
    CNTK has special optimizations to make this operation as efficient as an actual table lookup if the input is sparse.

    The lookup table in this layer is learnable,
    unless a user-specified one is supplied through the ``weights`` parameter.
    For example, to use an existing embedding table from a file in numpy format, use this::

      Embedding(weights=np.load('PATH.npy'))

    To initialize a learnable lookup table with a given numpy array that is to be used as
    the initial value, pass that array to the ``init`` parameter (not ``weights``).

    An ``Embedding`` instance owns its weight parameter tensor `E`, and exposes it as an attribute ``.E``.

    Example:
     >>> # learnable embedding
     >>> f = Embedding(5)
     >>> x = C.input_variable(3)
     >>> e = f(x)
     >>> e.shape
         (5,)
     >>> f.E.shape
         (3, 5)

     >>> # user-supplied embedding
     >>> f = Embedding(weights=[[.5, .3, .1, .4, .2], [.7, .6, .3, .2, .9]])
     >>> f.E.value
         array([[ 0.5,  0.3,  0.1,  0.4,  0.2],
                [ 0.7,  0.6,  0.3,  0.2,  0.9]], dtype=float32)
     >>> x = C.input_variable(2, is_sparse=True)
     >>> e = f(x)
     >>> e.shape
         (5,)
     >>> e(C.Value.one_hot([[1], [0], [0], [1]], num_classes=2))
     array([[ 0.7,  0.6,  0.3,  0.2,  0.9],
            [ 0.5,  0.3,  0.1,  0.4,  0.2],
            [ 0.5,  0.3,  0.1,  0.4,  0.2],
            [ 0.7,  0.6,  0.3,  0.2,  0.9]], dtype=float32)

    Args:
     shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
     init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): (learnable embedding only) initial value of weights `E`
     weights (NumPy array, mutually exclusive with ``init``, defuats to `None`): (user-supplied embedding only) the lookup table.
      The matrix rows are the embedding vectors, ``weights[i,:]`` being the embedding that corresponds to input category `i`.
     enable_weight_tying (bool): whether to produce both an input and output embedding for weight tying.
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the embedding operation to it
    '''

    if not C.is_default_override(init) and weights is not None:
        raise ValueError('Embedding: init and weights options are mutually exclusive')

    # parameters bound to this Function:
    # no weights given: learn the embedding
    if weights is None:
        if shape is None:
            raise ValueError('Embedding: output shape must be specified')
        init = get_default_override(Embedding, init=init)
        shape = _as_tuple(shape)
        weight_shape = _INFERRED + shape
        E = C.Parameter(weight_shape, init=init, name='E')
    # weights given: use them as constant
    else:
        import numpy as np
        weights = np.array(weights)
        weight_shape = np.shape(weights)
        if shape is not None:  # user may give shape, then it must match
            raise ValueError('Embedding: output shape must not be specified when weights are given')
        E = C.Constant(weights, name='E')

    # expression
    @C.BlockFunction('Embedding', name)
    def embed(x):
        return C.times(x, E)

    # expression
    @C.BlockFunction('TransposeEmbedding', name)
    def transpose_embed(x):
        return C.times_transpose(x, E)

    if enable_weight_tying:
        return embed, transpose_embed

    return embed


def QRNN(window: int = 1, hidden_dim=None, activation=C.tanh, return_full_state=False,
         variational_dropout_rate_input=None, variational_dropout_rate_output=None, name=''):
    """
    Quasi-Recurrent Neural Networks layer

    This is the CNTK implementation of [Salesforce Research](https://einstein.ai/)'s
    [Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576) paper.

    More details on tuning and application can be found in this paper:
    [An Analysis of Neural Language Modeling at Multiple Scales](https://arxiv.org/abs/1803.08240)

    QRNN is used in hangwriting recognition in Gboard too. More details in following link:
    https://ai.googleblog.com/2019/03/rnn-based-handwriting-recognition-in.html

    From the authors:
        The QRNN provides similar accuracy to the LSTM but can be between
        2 and 17 times faster than the highly optimized NVIDIA cuDNN LSTM
        implementation depending on the use case.
        If you use this code or our results in your research, please cite:
        @article{bradbury2016quasi,
          title={{Quasi-Recurrent Neural Networks}},
          author={Bradbury, James and Merity, Stephen and Xiong, Caiming and Socher, Richard},
          journal={International Conference on Learning Representations (ICLR 2017)},
          year={2017}
        }

    Examples:
        input_tensor = C.sequence.input_variable(input_dim)

        hidden = QRNN(window=2, hidden_dim=hidden_dim)(input_tensor)
        prediction = Dense(1)(C.sequence.last(hidden))

    Arguments:
        window (`int`):  Defines the size of the convolutional window (how many previous
          tokens to look when computing the QRNN values). Defaults 1.
        hidden_dim (int): size of hidden dim of h, c and o
        activation: cell activation function
        return_full_state: if to return cell and hidden states. Default false.
        name: name of function instance in network

    Returns:
        :class:`~cntk.ops.functions.Function`: OR
        tuple of :class:`~cntk.ops.functions.Function`:

    """

    sequential_conv = SequentialConvolution(filter_shape=(window,),
                                            num_filters=3 * hidden_dim,
                                            pad=False,
                                            reduction_rank=1,
                                            name='conv')

    @C.Function
    def f_pool(c, zf):
        z = C.slice(zf, 0, 0, hidden_dim)
        f = C.slice(zf, 0, hidden_dim, 2 * hidden_dim)
        return f * c + (1 - f) * z

    @C.BlockFunction('QRNN', name)
    def model(input_tensor):

        input_sequence = input_tensor
        if window > 1:
            # to ensure causal relation is still preserved
            input_sequence = Cx.sequence.pad(input_sequence, (window - 1, 0), constant_value=0)

        gate_values = sequential_conv(input_sequence)

        x = C.slice(gate_values, -1, 0, hidden_dim)
        forget = C.slice(gate_values, -1, hidden_dim, 2 * hidden_dim)
        output = C.slice(gate_values, -1, 2 * hidden_dim, 3 * hidden_dim)

        z = activation(x)
        f = C.sigmoid(forget)
        o = C.sigmoid(output)

        # Pooling
        zf = C.splice(z, f)
        c = Cx.layers.Recurrence(f_pool,
                                 dropout_rate_input=variational_dropout_rate_input,
                                 dropout_rate_output=variational_dropout_rate_output)(zf)
        h = o * c  # o pool

        if return_full_state:
            return h, c
        else:
            return h

    return model


def SinusoidalPositionalEmbedding(dim, min_timescale=1.0, max_timescale=1.0e4, name=''):
    """ Gets a bunch of sinusoids of different frequencies and add it to the input sequence

    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase. This allows attention to learn to use absolute and relative positions.

    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention. The use of relative position is possible because
    sin(x+y) and cos(x+y) can be expressed in terms of y, sin(x) and cos(x).

    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale. The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need" in
    that if input_dim is odd, the last dim will be a zero value.

    This implementation is equivalent to get_timing_signal_1d() in tensorflow's tensor2tensor:
        https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/
          tensor2tensor/layers/common_attention.py

    There are no learnable parameters in this embedding.

    Example:
        import cntk as C
        import cntkx as Cx

        a = C.sequence.input_variable(10)
        b = Cx.layers.SinusoidalPositionalEmbedding(10)(a)

        assert b.shape == (10, )

    Arguments:
        dim (int): dimension of embedding (typically must be the same as the incoming tensor to be embedded)
        min_timescale (float): geometric sequence of timescales starting with min_timescale
        max_timescale (float): geometric sequence of timescales ending with max_timescale
        name (str): a name for this layer.

    Returns:
        :class:`~cntk.ops.functions.Function`: same shape as input sequence tensor

    """

    @C.BlockFunction('SinusoidalPositionalEmbedding', name)
    def embedding(x):
        num_timescales = dim // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1))
        inv_timescales = C.constant(min_timescale * np.exp(np.arange(num_timescales) * -log_timescale_increment),
                                    dtype=np.float32)

        pos = Cx.sequence.position(x)  # pos: [#, *] [1, ]
        scaled_time = pos * inv_timescales  # scaled_time: [#, *] [num_timescales,]

        s = C.sin(scaled_time)
        c = C.cos(scaled_time)
        signal = C.splice(s, c)

        # last dim gets a 0 value if input_dim is odd
        if dim % 2 != 0:
            signal = C.pad(signal, [[0, 1]])

        return signal

    return embedding


# Sequential Convolution -- create a sequential convolution layer with optional non-linearity
# This is the newer version that supports ND sequential convolution with arbitrary strides.
#             ( (sample shape) +  (output shape) +  (reduction shape) + (spatial shape)   )
#    in     : ( (sample shape) +                 +  (reduction shape) + (spatial shape)   )
#    kernel : (                +  (output shape) +  (reduction shape) + (rec field shape) )
#    out    : ( (sample shape) +  (output shape) +                    + (spatial shape)   )
def SequentialConvolution(filter_shape,     # shape of receptive field, e.g. (3,3). filter_shape[0] is for sequence axis.
                          num_filters=None, # e.g. 64 or None (which means 1 channel and don't add a dimension)
                          activation=default_override_or(identity),
                          init=default_override_or(C.glorot_uniform()),
                          pad=default_override_or(False),
                          strides=1,
                          sharing=True,     # (must be True currently)
                          bias=default_override_or(True),
                          init_bias=default_override_or(0),
                          reduction_rank=1, # (0 means input has no depth dimension, e.g. audio signal or B&W image)  --TODO: call it item_rank?
                          transpose_weight=False,  # (must be False currently)
                          dilation=1,
                          groups=1,
                          input_num_filters=None,
                          max_temp_mem_size_in_samples=0,
                          name=''):
    '''
    SequentialConvolution(filter_shape, num_filters=None, activation=identity, init=glorot_uniform(), pad=False, strides=1, sharing=True, bias=True, init_bias=0, reduction_rank=1, transpose_weight=False, dilation=1, groups=1, max_temp_mem_size_in_samples=0, op_name='Convolution', name='')

    This implementation allows for (1) group convolution and (2) initialisation with reduction rank = 1, both of which
    was not possible in the original cntk implementayion.

    More details please refer to the original cntk documentation.

    Args:
     filter_shape (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
     num_filters (int, defaults to `None`): number of filters (output feature-map depth), or ``()`` to denote scalar output items (output shape will have no depth axis).
     activation (:class:`~cntk.ops.functions.Function`, defaults to `identity`): optional function to apply at the end, e.g. `relu`
     init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
     pad (`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the filter will be shifted over the "valid"
      area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
      the filter will be applied to all input positions, and positions outside the valid region will be considered containing zero.
      Use a `tuple` to specify a per-axis value.
     strides (`int` or `tuple` of `ints`, defaults to 1): stride of the convolution (increment when sliding the filter over the input). Use a `tuple` to specify a per-axis value.
     sharing (bool, defaults to `True`): When `True`, every position uses the same Convolution kernel.  When `False`, you can have a different Convolution kernel per position, but `False` is not supported.
     bias (bool, optional, defaults to `True`): the layer will have no bias if `False` is passed here
     init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
     reduction_rank (`int`, defaults to 1): set to 0 if input items are scalars (input has no depth axis), e.g. an audio signal or a black-and-white image
      that is stored with tensor shape (H,W) instead of (1,H,W)
     transpose_weight (bool, defaults to `False`): When this is `True` this is convolution, otherwise this is correlation (which is common for most toolkits)
     dilation (tuple, optional): the dilation value along each axis, default 1 mean no dilation.
     groups (`int`, default 1): number of groups during convolution, that controls the connections between input and output channels. Deafult value is 1,
      which means that all input channels are convolved to produce all output channels. A value of N would mean that the input (and output) channels are
      divided into N groups with the input channels in one group (say i-th input group) contributing to output channels in only one group (i-th output group).
      Number of input and output channels must be divisble by value of groups argument. Also, value of this argument must be strictly positive, i.e. groups > 0.
     input_num_filters (int): used for group convolution
     max_temp_mem_size_in_samples (int, defaults to 0): Limits the amount of memory for intermediate convolution results.  A value of 0 means, memory is automatically managed.
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the sequential convolution operation to it
    '''

    activation = C.get_default_override(SequentialConvolution, activation=activation)
    init       = C.get_default_override(SequentialConvolution, init=init)
    pad        = C.get_default_override(SequentialConvolution, pad=pad)
    bias       = C.get_default_override(SequentialConvolution, bias=bias)
    init_bias  = C.get_default_override(SequentialConvolution, init_bias=init_bias)

    # tuplify all tuple inputs that can also be given as scalars if rank 1
    filter_shape = _as_tuple(filter_shape)
    num_filters  = _as_tuple(num_filters or ())
    filter_rank  = len(filter_shape)
    strides      = _pad_to_shape(filter_shape, strides, 'strides')
    sharing      = _pad_to_shape(filter_shape, sharing, 'sharing')
    pad          = _pad_to_shape(filter_shape, pad, 'pad')
    dilation     = _pad_to_shape(filter_shape, dilation, 'dilation')

    if (reduction_rank != 0) and (reduction_rank != 1):
        raise NotImplementedError("Convolution: reduction_rank must be 0 or 1")
    if transpose_weight:
        raise NotImplementedError("Convolution: transpose_weight option currently not supported")
    if not sharing:
        raise NotImplementedError("Convolution: sharing option currently must be True")
    if groups <= 0:
        raise ValueError("Convolution: groups must be strictly positive, i.e. groups > 0.")
    if input_num_filters and input_num_filters % groups != 0:
        raise ValueError('input_num_filters must be divisible by number of groups')
    if groups > 1 and num_filters[0] % groups != 0:
        raise ValueError('num_filters must be divisible by number of groups')
    if groups > 1 and reduction_rank == 0:
        raise ValueError('reduction_rank cannot be zero in group convolution i.e. there must be incoming channels')

    num_filters_per_group = None
    if input_num_filters and num_filters[0] % groups == 0 and input_num_filters % groups == 0:
        num_filters_per_group = (int(input_num_filters / groups), )
        # TODO: work on groups, understand how reduction==0 and init=np might affect group which doesn't have inferred

    # The convolution() function currently requires exactly one input and one output depth axis.
    # So we emulate those dimensions on this level.
    emulating_output_depth = num_filters == ()
    emulating_input_depth  = reduction_rank == 0

    actual_output_channels_shape = num_filters if not emulating_output_depth else (1,)
    actual_reduction_shape       = _INFERRED if num_filters_per_group is None else num_filters_per_group
    actual_filter_shape          = filter_shape

    # add the dimension to the options as well
    num_emulated_axes = emulating_input_depth
    strides = (1,)     * num_emulated_axes + strides
    sharing = (True,)  * num_emulated_axes + sharing
    pad     = (False,) * num_emulated_axes + pad

    kernel_shape = actual_reduction_shape + actual_filter_shape  # kernel := filter plus reductionDims

    # init can be an np.array, which must have the correct dimensions subject to faking depth
    # Once we no longer fake depth at this outer level, we can remove this.
    if isinstance(init, np.ndarray):

        if init.shape[-len(filter_shape):] != filter_shape and init.shape[0] != num_filters[0]:
            raise ValueError(f"a constant initializer was passed that is of wrong shape {init.shape}")

        init_kernel = init

        # with no inferred axis in W and no emulated axis,
        # padding must be explicit for all axes (channel, seq, static axes)
        # typically, with inferred axis, pad omits channel pad, which is taken to be False. (seq, static axes)
        # with emulate axis, the extra pad would have been supplied already
        pad = (False, ) * reduction_rank + pad  # assume pad[0] is seq axis, pad[1:] is static axes

    elif num_filters_per_group:

        # with no inferred axis in W and no emulated axis,
        # padding must be explicit for all axes (channel, seq, static axes)
        # typically, with inferred axis, pad omits channel pad, which is taken to be False. (seq, static axes)
        # with emulate axis, the extra pad would have been supplied already
        pad = (False,) * reduction_rank + pad  # assume pad[0] is seq axis, pad[1:] is static axes
        init_kernel = _initializer_for(init, Record(filter_rank=filter_rank, output_rank=-len(actual_output_channels_shape)))

    else:
        init_kernel = _initializer_for(init, Record(filter_rank=filter_rank, output_rank=-len(actual_output_channels_shape)))

    # parameters bound to this Function
    # For sequential we must reduce bias filter rank by 1, as we get the rank from kernel filter shape,
    # and that contains the seq axis which should be omitted.
    bias_filter_rank = len(actual_filter_shape) - 1
    W = C.Parameter(actual_output_channels_shape + kernel_shape,            init=init_kernel, name='W')                   # (K, C, W, H) aka [ H x W x C x K ]
    b = C.Parameter(actual_output_channels_shape + (1,) * bias_filter_rank, init=init_bias,   name='b') if bias else None # (K,    1, 1) aka [ 1 x 1 x     K ]
    # print(W.shape)

    # expression
    @C.BlockFunction('SequentialConvolution', name)
    def convolve(x):
        # insert additional axes for various purposes
        if reduction_rank == 0:  # only ever going to be 1 when reduction rank = 0
            # x: (spatial_shape)
            x = C.expand_dims(x, axis=C.Axis.new_leading_axis())  # e.g. (480, 640) -> (1, 480, 640)
            # x: (in_depth or emulated_in_depth, spatial_shape)

        # actual convolution
        r = C.convolution(W, x,
                          strides=strides,
                          sharing=sharing,
                          auto_padding=pad,
                          sequential=True,
                          dilation=dilation,
                          groups=groups,
                          max_temp_mem_size_in_samples=max_temp_mem_size_in_samples)

        if bias:
            r = r + b

        # if no output dimension is desired, then strip it
        # also need to strip the fake singleton axes, since they are not reduced away
        num_axes_to_remove = emulating_output_depth
        if num_axes_to_remove > 0:
            # (out_depth, emulated axes, spatial_shape)
            r = C.squeeze(r)  # e.g. (2000, 1, 480, 640) -> (2000, 480, 640)
            # (out_depth, spatial_shape)

        if activation is not None:
            r = activation(r)

        return r

    return convolve


def Convolution(filter_shape,      # shape of receptive field, e.g. (3,3)
                num_filters=None,  # e.g. 64 or None (which means 1 channel and don't add a dimension)
                sequential=False,  # time convolution if True (filter_shape[0] corresponds to dynamic axis)
                activation=default_override_or(identity),
                init=default_override_or(C.glorot_uniform()),
                pad=default_override_or(False),
                strides=1,
                sharing=True,      # (must be True currently)
                bias=default_override_or(True),
                init_bias=default_override_or(0),
                reduction_rank=1,  # (0 means input has no depth dimension, e.g. audio signal or B&W image)  --TODO: call it item_rank?
                transpose_weight=False,  # (must be False currently)
                dilation=1,
                groups=1,
                input_num_filters=None,
                max_temp_mem_size_in_samples=0,
                op_name='Convolution',
                name=''):
    '''
    Convolution(filter_shape, num_filters=None, sequential=False, activation=identity, init=glorot_uniform(), pad=False, strides=1, sharing=True, bias=True, init_bias=0, reduction_rank=1, transpose_weight=False, dilation=1, groups=1, max_temp_mem_size_in_samples=0, op_name='Convolution', name='')

    This implementation allows for (1) group convolution and (2) initialisation with reduction rank = 1, both of which
    was not possible in the original cntk implementayion.

    More details please refer to the original cntk documentation.

    Args:
     filter_shape (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
     num_filters (int, defaults to `None`): number of filters (output feature-map depth), or ``()`` to denote scalar output items (output shape will have no depth axis).
     sequential (bool, defaults to `False`): if `True`, also convolve along the dynamic axis. ``filter_shape[0]`` corresponds to dynamic axis.
     activation (:class:`~cntk.ops.functions.Function`, defaults to `identity`): optional function to apply at the end, e.g. `relu`
     init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
     pad (`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the filter will be shifted over the "valid"
      area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
      the filter will be applied to all input positions, and positions outside the valid region will be considered containing zero.
      Use a `tuple` to specify a per-axis value.
     strides (`int` or `tuple` of `ints`, defaults to 1): stride of the convolution (increment when sliding the filter over the input). Use a `tuple` to specify a per-axis value.
     sharing (bool, defaults to `True`): When `True`, every position uses the same Convolution kernel.  When `False`, you can have a different Convolution kernel per position, but `False` is not supported.
     bias (bool, optional, defaults to `True`): the layer will have no bias if `False` is passed here
     init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
     reduction_rank (`int`, defaults to 1): set to 0 if input items are scalars (input has no depth axis), e.g. an audio signal or a black-and-white image
      that is stored with tensor shape (H,W) instead of (1,H,W)
     transpose_weight (bool, defaults to `False`): When this is `True` this is convolution, otherwise this is correlation (which is common for most toolkits)
     dilation (tuple, optional): the dilation value along each axis, default 1 mean no dilation.
     groups (`int`, default 1): number of groups during convolution, that controls the connections between input and output channels. Deafult value is 1,
      which means that all input channels are convolved to produce all output channels. A value of N would mean that the input (and output) channels are
      divided into N groups with the input channels in one group (say i-th input group) contributing to output channels in only one group (i-th output group).
      Number of input and output channels must be divisble by value of groups argument. Also, value of this argument must be strictly positive, i.e. groups > 0.
     max_temp_mem_size_in_samples (int, defaults to 0): Limits the amount of memory for intermediate convolution results.  A value of 0 means, memory is automatically managed.
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the convolution operation to it
    '''

    activation = C.get_default_override(Convolution, activation=activation)
    init       = C.get_default_override(Convolution, init=init)
    pad        = C.get_default_override(Convolution, pad=pad)
    bias       = C.get_default_override(Convolution, bias=bias)
    init_bias  = C.get_default_override(Convolution, init_bias=init_bias)

    # tuplify all tuple inputs that can also be given as scalars if rank 1
    filter_shape = _as_tuple(filter_shape)
    num_filters  = _as_tuple(num_filters or ())
    filter_rank  = len(filter_shape)
    strides      = _pad_to_shape(filter_shape, strides, 'strides')
    sharing      = _pad_to_shape(filter_shape, sharing, 'sharing')
    pad          = _pad_to_shape(filter_shape, pad, 'pad')
    dilation     = _pad_to_shape(filter_shape, dilation, 'dilation')

    if (reduction_rank != 0) and (reduction_rank != 1):
        raise NotImplementedError("Convolution: reduction_rank must be 0 or 1")
    if transpose_weight:
        raise NotImplementedError("Convolution: transpose_weight option currently not supported")
    if not sharing:
        raise NotImplementedError("Convolution: sharing option currently must be True")
    if groups <= 0:
        raise ValueError("Convolution: groups must be strictly positive, i.e. groups > 0.")
    if input_num_filters and input_num_filters % groups != 0:
        raise ValueError('input_num_filters must be divisible by number of groups')
    if groups > 1 and num_filters[0] % groups != 0:
        raise ValueError('num_filters must be divisible by number of groups')
    if groups > 1 and reduction_rank == 0:
        raise ValueError('reduction_rank cannot be zero in group convolution i.e. there must be incoming channels')
    if sequential:
        raise ValueError("Use cntk.layers.SequentialConvolution instead")

    num_filters_per_group = None
    if input_num_filters and num_filters[0] % groups == 0 and input_num_filters % groups == 0:
        num_filters_per_group = (int(input_num_filters / groups),)
        # TODO: work on groups, understand how reduction==0 and init=np might affect group which doesn't have inferred

    # The convolution() function currently requires exactly one input and one output depth axis.
    # So we emulate those dimensions on this level
    emulating_output_depth = num_filters == ()
    emulating_input_depth = reduction_rank == 0

    actual_output_channels_shape = num_filters if not emulating_output_depth else (1,)
    actual_reduction_shape = _INFERRED if num_filters_per_group is None else num_filters_per_group
    actual_filter_shape = filter_shape

    # add the dimension to the options as well
    num_emulated_axes = emulating_input_depth
    strides = (1,) * num_emulated_axes + strides
    sharing = (True,) * num_emulated_axes + sharing
    pad = (False,) * num_emulated_axes + pad

    kernel_shape = actual_reduction_shape + actual_filter_shape  # kernel := filter plus reductionDims

    # init can be an np.array, which must have the correct dimensions subject to faking depth
    # Once we no longer fake depth at this outer level, we can remove this.
    if isinstance(init, np.ndarray):

        if init.shape[-len(filter_shape):] != filter_shape and init.shape[0] != num_filters[0]:
            raise ValueError("a constant initializer was passed that is of wrong shape")

        init_kernel = init

        # with no inferred axis in W and no emulated axis,
        # padding must be explicit for all axes (channel, static axes)
        # typically, with inferred axis, pad omits channel pad, which is taken to be False. (static axes)
        # with emulate axis, the extra pad would have been supplied already
        pad = (False,) * reduction_rank + pad

    elif num_filters_per_group:

        # with no inferred axis in W and no emulated axis,
        # padding must be explicit for all axes (channel, seq, static axes)
        # typically, with inferred axis, pad omits channel pad, which is taken to be False. (seq, static axes)
        # with emulate axis, the extra pad would have been supplied already
        pad = (False,) * reduction_rank + pad  # assume pad[0] is seq axis, pad[1:] is static axes
        init_kernel = _initializer_for(init, Record(filter_rank=filter_rank, output_rank=-len(actual_output_channels_shape)))

    else:
        init_kernel = _initializer_for(init, Record(filter_rank=filter_rank, output_rank=-len(actual_output_channels_shape)))

    # parameters bound to this Function
    W = C.Parameter(actual_output_channels_shape + kernel_shape,                    init=init_kernel, name='W')                    # (K, C, H, W) aka [ W x H x C x K ]
    b = C.Parameter(actual_output_channels_shape + (1,) * len(actual_filter_shape), init=init_bias,   name='b') if bias else None  # (K,    1, 1) aka [ 1 x 1 x     K ]

    # expression
    @C.BlockFunction(op_name, name)
    def convolve(x):
        # insert additional axis to emulate depth
        if reduction_rank == 0:
            # x: (spatial_shape)
            x = C.expand_dims(x, axis=C.Axis.new_leading_axis())  # e.g. (480, 640) -> (1, 480, 640)
            # x: (in_depth or emulated_in_depth, spatial_shape)

        # actual convolution
        r = C.convolution(W, x,
                          strides=strides,
                          sharing=sharing,
                          auto_padding=pad,
                          dilation=dilation,
                          groups=groups,
                          max_temp_mem_size_in_samples=max_temp_mem_size_in_samples)

        if bias:
            r = r + b

        # if no output dimension is desired, then strip it
        # also need to strip the fake singleton axes, since they are not reduced away
        num_axes_to_remove = emulating_output_depth
        if num_axes_to_remove > 0:
            # (out_depth, emulated axes, spatial_shape)
            r = C.squeeze(r)  # e.g. (2000, 1, 480, 640) -> (2000, 480, 640)
            # (out_depth, spatial_shape)

        if activation is not None:
            r = activation(r)
        return r

    return convolve


def Convolution2D(filter_shape,     # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
                  num_filters=None, # e.g. 64 or None (which means 1 channel and don't add a dimension)
                  activation=default_override_or(identity),
                  init=default_override_or(C.glorot_uniform()),
                  pad=default_override_or(False),
                  strides=1,
                  bias=default_override_or(True),
                  init_bias=default_override_or(0),
                  reduction_rank=1, # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                  dilation=1,
                  groups=1,
                  input_num_filters=None,
                  name=''):
    '''
    Convolution2D(filter_shape, num_filters=None, activation=identity, init=glorot_uniform(), pad=False, strides=1, bias=True, init_bias=0, reduction_rank=1, name='')

    Layer factory function to create a 2D convolution layer with optional non-linearity.
    Same as `Convolution()` except that filter_shape is verified to be 2-dimensional.
    See `Convolution()` for extensive documentation.

    Args:
     filter_shape (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
     num_filters (int, defaults to `None`): number of filters (output feature-map depth), or ``()`` to denote scalar output items (output shape will have no depth axis).
     activation (:class:`~cntk.ops.functions.Function`, defaults to `identity`): optional function to apply at the end, e.g. `relu`
     init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
     pad (`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the filter will be shifted over the "valid"
      area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
      the filter will be applied to all input positions, and positions outside the valid region will be considered containing zero.
      Use a `tuple` to specify a per-axis value.
     strides (`int` or `tuple` of `ints`, defaults to 1): stride of the convolution (increment when sliding the filter over the input). Use a `tuple` to specify a per-axis value.
     bias (bool, defaults to `True`): the layer will have no bias if `False` is passed here
     init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
     reduction_rank (`int`, defaults to 1): set to 0 if input items are scalars (input has no depth axis), e.g. an audio signal or a black-and-white image
      that is stored with tensor shape (H,W) instead of (1,H,W)
     dilation (tuple, optional): the dilation value along each axis, default 1 mean no dilation.
     groups (`int`, default 1): number of groups during convolution, that controls the connections between input and output channels. Deafult value is 1,
      which means that all input channels are convolved to produce all output channels. A value of N would mean that the input (and output) channels are
      divided into N groups with the input channels in one group (say i-th input group) contributing to output channels in only one group (i-th output group).
      Number of input and output channels must be divisble by value of groups argument. Also, value of this argument must be strictly positive, i.e. groups > 0.
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the convolution operation to it

    '''

    activation = C.get_default_override(Convolution2D, activation=activation)
    init       = C.get_default_override(Convolution2D, init=init)
    pad        = C.get_default_override(Convolution2D, pad=pad)
    bias       = C.get_default_override(Convolution2D, bias=bias)
    init_bias  = C.get_default_override(Convolution2D, init_bias=init_bias)
    if len(_as_tuple(filter_shape)) > 2:
         raise ValueError('Convolution2D: filter_shape must be a scalar or a 2D tuple, e.g. 3 or (3,3)')
    filter_shape = _pad_to_shape((0, 0), filter_shape, 'filter_shape')

    return Convolution(filter_shape, num_filters=num_filters, activation=activation, init=init, pad=pad, sequential=False,
                       strides=strides, sharing=True, bias=bias, init_bias=init_bias, reduction_rank=reduction_rank,
                       dilation=dilation, groups=groups, input_num_filters=input_num_filters, op_name='Convolution2D',
                       name=name)


def Conv2DMaxPool(n, conv_filter_shape,  # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
                  pool_filter_shape,  # shape of receptive field, e.g. (3,3)
                  conv_num_filters=None,  # e.g. 64 or None (which means 1 channel and don't add a dimension)
                  activation=default_override_or(identity),
                  init=default_override_or(C.glorot_uniform()),
                  conv_pad=default_override_or(False),
                  conv_strides=1,
                  bias=default_override_or(True),
                  init_bias=default_override_or(0),
                  reduction_rank=1,  # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                  dilation=1,
                  groups=1,
                  input_num_filters=None,
                  pool_strides=1,
                  pool_pad=default_override_or(False),
                  name_prefix=''):
    """ Stack of Convolution 2D followed by one max pooling layer. Convenience wrapper. """

    conv_stack = Convolution2DStack(n, conv_filter_shape, conv_num_filters, activation, init, conv_pad, conv_strides,
                                    bias, init_bias, reduction_rank, dilation, groups, input_num_filters, name_prefix)

    maxpool = MaxPooling(pool_filter_shape, pool_strides, pool_pad, name_prefix + '_pool')

    def layer(x):
        x = conv_stack(x)
        x = maxpool(x)
        return x

    return layer


def Convolution2DStack(num_conv_layers,  # num of convolutional layers in the stack
                       filter_shape,  # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
                       num_filters=None,  # e.g. 64 or None (which means 1 channel and don't add a dimension)
                       activation=default_override_or(identity),
                       init=default_override_or(C.glorot_uniform()),
                       pad=default_override_or(False),
                       strides=1,
                       bias=default_override_or(True),
                       init_bias=default_override_or(0),
                       reduction_rank=1,  # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                       dilation=1,
                       groups=1,
                       input_num_filters=None,
                       name_prefix=''):
    """ A stack of of convolutional layers. Convenience wrapper. """

    convs = [Convolution2D(filter_shape, num_filters, activation, init, pad, strides, bias,
                           init_bias, reduction_rank, dilation, groups, input_num_filters,
                           name_prefix + f'_conv_{i}') for i in range(num_conv_layers)]

    def inner(x):

        for conv in convs:
            x = conv(x)

        return x

    return inner


def SpatialPyramidPooling(bins: tuple, name=''):
    """ Spatial pyramid pooling layer for 2D inputs.

    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun (https://arxiv.org/abs/1406.4729)

    SSP is used for multi-sized training where during training we implement the varying-input-size SPP-net
    by two fixed-size networks that share parameters. SSP layer will be different for the 2 network that
    shares parameters since the SSP would have different windows and stride.

    The final output shape would be input_num_filters * reduce_sum(square(bins))
    e.g. bins = (1, 3, 5) and input_num_filters = 32 then output_shape = (32 * (1 * 1 + 3 * 3 + 5 * 5), ) regardless
    of input feature map's spatial dimension.

    Arguments:
        bins (tuple): tuple of ints stating the depth of the pyramid and number of bins at each level.
        name (str, optional): name of layer

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """

    def spp(x):
        spatial = x.shape[1:]
        filter_shapes = [tuple(math.ceil(s / bin) for s in spatial) for bin in bins]
        strides = [tuple(math.floor(s / bin) for s in spatial) for bin in bins]

        pools = [MaxPooling(filter_shape, stride, pad=False) for filter_shape, stride in zip(filter_shapes, strides)]
        features = [C.flatten(pool(x)) for pool in pools]
        return C.squeeze(C.splice(*features), name=name)

    return spp


def SequentialMaxPooling(filter_shape,  # shape of receptive field, e.g. (3,3). filter_shape[0] is for sequence axis.
                         strides=1,     # strides[0] is for sequence axis.
                         pad=default_override_or(True),   # pad[0] is for sequence axis.
                         name=''):
    """ Layer factory function to create a max-pooling layer that works with sequences

    Sequential max pooling has a slight bug in that even when Pad=False, sequence axis will still be
    padded and asymmetrically padded so on the right. i.e. there may be an extrac sequence element. But it should
    not be an issue since error in border pixels typically wouldn't affect results.

    Example:
        # rgb image of height 25 and variable width
        a = C.sequence.input_variable((3, 25))

        # max pool (2,2) in height and width with stride (2,2) in height and width, no padding
        b = SequentialMaxPooling(filter_shape=(2, 2), strides=(2, 2), pad=False)(a)
        assert b.shape == (3, 12)

        # max pool (2,2) in height and width with stride (2,2) in height and width, with padding
        b = SequentialMaxPooling(filter_shape=(2, 2), strides=(2, 2), pad=True)(a)
        assert b.shape == (3, 13)


    Arguments:
        filter_shape (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
        strides (`int` or `tuple` of `ints`, defaults to 1): stride (increment when sliding over the input). Use a `tuple` to specify a per-axis value.
        pad (`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the pooling operation will be shifted over the "valid"
          area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
          pooling will be applied to all input positions, and positions outside the valid region will be considered containing zero.
          Use a `tuple` to specify a per-axis value.
        name (str, defaults to ''): the name of the function instance in the network


    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the max-pooling operation to it

    """
    assert isinstance(filter_shape, tuple), "filter must be a tuple"
    pad = get_default_override(SequentialConvolution, pad=pad)

    if not isinstance(pad, tuple):
        pad = tuple(pad for __ in filter_shape)

    if not isinstance(strides, tuple):
        strides = tuple(strides for __ in filter_shape)

    # for pooling in static axes
    pool_filter_shape = filter_shape[1:]
    pool_pad = pad[1:]
    pool_strides = strides[1:]

    # static_pool over (channel_static_axis, height_static_axis)
    if pool_filter_shape and pool_strides and pool_pad:

        static_pooler = MaxPooling(filter_shape=filter_shape[1:], strides=pool_strides, pad=pool_pad, name='static_pooler')

    else:

        static_pooler = identity  # when there is no static axes to pool

    @C.BlockFunction('SequentialMaxPooling', name)
    def inner(x):
        if pad[0]:  # sequential axis
            # when kernel is even, padding will be asymmetric in left and right
            right_pad = int((filter_shape[0] - 1) / 2) if filter_shape[0] % 2 else int(filter_shape[0] / 2)
            left_pad = right_pad if filter_shape[0] % 2 else right_pad - 1

            past = [C.sequence.past_value(x, time_step=i + 1) for i in range(left_pad)]
            future = [C.sequence.future_value(x, time_step=i + 1) for i in range(right_pad)]

            past_now_future = past + [x] + future

        else:

            future = [C.sequence.future_value(x, time_step=i + 1) for i in range(filter_shape[1] - 1)]
            past_now_future = [x] + future

        windows = C.splice(*past_now_future, axis=C.Axis.new_leading_axis())
        # windows: [#, *] [concat, channel, static_axes...]

        selected_windows = Cx.sequence.stride(windows, strides[0])
        # selected_windows: [#, **] [concat, channel, static_axes...]
        # assert windows.shape == selected_windows.shape

        # Pooling between sequential elements done by reduce_max on windows
        # BUGBUG: do not set keepdims=False in reduce_max, will raise error
        sequential_max_pooled = C.squeeze(C.reduce_max(selected_windows, axis=0), axes=0)
        # sequential_max_pooled: [#, **] [channel, static_axes...]

        pooled = static_pooler(sequential_max_pooled)
        # sequential_max_pooled: [#, **] [channel, pooled_static_axes...]

        return pooled

    return inner


def SequentialAveragePooling(filter_shape,  # shape of receptive field, e.g. (3,3) filter_shape[0] is for sequence axis
                             strides=1,  # strides[0] is for sequence axis.
                             pad=default_override_or(True),  # pad[0] is for sequence axis.
                             name=''):
    """ Layer factory function to create a average-pooling layer that works with sequences

    Sequential average pooling has a slight bug in that even when Pad=False, sequence axis will still be
    padded and asymmetrically padded so on the right. i.e. there may be an extrac sequence element. But it should
    not be an issue since error in border pixels typically wouldn't affect results.

    Note that this implementation includes padding as part of the average, different from the standard average pooling
    in tensorflow, cntk and pytorch.

    For a corner average pool with kernel of (5, 5):

        0 0 0  0  0
        0 0 0  0  0
        0 0 1  2  3
        0 0 6  7  8
        0 0 11 12 13

        Sum = (1+2+3+6+7+8+11+12+13) = 63
        SequentialAveragePooling: 63/25 = 2.52
        AveragePooling (cntk, tf, pytorch)  : 63/9  = 7

    Example:
        # rgb image of height 25 and variable width
        a = C.sequence.input_variable((3, 25))

        # max pool (2,2) in height and width with stride (2,2) in height and width, no padding
        b = SequentialAveragePooling(filter_shape=(2, 2), strides=(2, 2), pad=False)(a)
        assert b.shape == (3, 12)

        # max pool (2,2) in height and width with stride (2,2) in height and width, with padding
        b = SequentialAveragePooling(filter_shape=(2, 2), strides=(2, 2), pad=True)(a)
        assert b.shape == (3, 13)


    Arguments:
        filter_shape (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
        strides (`int` or `tuple` of `ints`, defaults to 1): stride (increment when sliding over the input). Use a `tuple` to specify a per-axis value.
        pad (`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the pooling operation will be shifted over the "valid"
          area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
          pooling will be applied to all input positions, and positions outside the valid region will be considered containing zero.
          Use a `tuple` to specify a per-axis value.
        name (str, defaults to ''): the name of the function instance in the network


    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the average-pooling operation to it

    """
    assert isinstance(filter_shape, tuple), "filter must be a tuple"
    pad = get_default_override(SequentialConvolution, pad=pad)

    if not isinstance(pad, tuple):
        pad = tuple(pad for __ in filter_shape)

    if not isinstance(strides, tuple):
        strides = tuple(strides for __ in filter_shape)

    # for pooling in static axes
    pool_filter_shape = filter_shape[1:]
    pool_pad = pad[1:]
    pool_strides = strides[1:]

    # static_pool over (channel_static_axis, height_static_axis)
    if pool_filter_shape and pool_strides and pool_pad:

        static_pooler = AveragePooling(filter_shape=filter_shape[1:], strides=pool_strides, pad=pool_pad, name='static_pooler')

    else:

        static_pooler = identity  # when there is no static axes to pool

    @C.BlockFunction('SequentialAveragePooling', name)
    def inner(x):
        if pad[0]:  # sequential axis
            # when kernel is even, padding will be asymmetric in left and right
            right_pad = int((filter_shape[0] - 1) / 2) if filter_shape[0] % 2 else int(filter_shape[0] / 2)
            left_pad = right_pad if filter_shape[0] % 2 else right_pad - 1

            past = [C.sequence.past_value(x, time_step=i + 1) for i in range(left_pad)]
            future = [C.sequence.future_value(x, time_step=i + 1) for i in range(right_pad)]

            past_now_future = past + [x] + future

        else:

            future = [C.sequence.future_value(x, time_step=i + 1) for i in range(filter_shape[1] - 1)]
            past_now_future = [x] + future

        windows = C.splice(*past_now_future, axis=C.Axis.new_leading_axis())
        # windows: [#, *] [concat, channel, static_axes...]

        selected_windows = Cx.sequence.stride(windows, strides[0])
        # selected_windows: [#, **] [concat, channel, static_axes...]
        # assert windows.shape == selected_windows.shape

        # Pooling between sequential elements done by reduce_mean on windows
        # BUGBUG: do not set keepdims=False in reduce_max, will raise error
        sequential_ave_pooled = C.squeeze(C.reduce_mean(selected_windows, axis=0), axes=0)
        # sequential_ave_pooled: [#, **] [channel, static_axes...]

        pooled = static_pooler(sequential_ave_pooled)
        # pooled: [#, **] [channel, pooled_static_axes...]

        return pooled

    return inner


def GatedLinearUnit(window=2, hidden_dim=None, activation=C.sigmoid, name=''):
    """
    Gated Linear Unit or gated convolutional neural network is a finite context approach
    through stacked convolutions, which can be  more  efficient  since  they  allow
    parallelization over sequential tokens.

    Context is captured through the stacking multiple gated linear units with window size more than one unlike
    in QRNN where there is still an explicit recurrence/pooling relationship temporally.

    Example:
        a = C.sequence.input_variable(56)
        b = Cx.layers.GatedLinearUnit(2, 100)(a)

        assert b.shape == (100, )

    Arguments:
        window (`int`):  Defines the size of the convolutional window (how many previous
          tokens to look when computing the gated linear unit values). Defaults 2.
        hidden_dim (int): size of hidden output dim. Must be divisible by 2.
        activation (`~cntk.ops.functions.Function`): gate function
        name (str): name of function instance in network

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    assert hidden_dim % 2 == 0, "hidden dimension must be divisible by 2"
    linear = SequentialConvolution(filter_shape=(window,), num_filters=2 * hidden_dim, pad=False)

    @C.BlockFunction('GatedLinearUnit', name)
    def inner(input_tensor):

        input_sequence = input_tensor
        if window > 1:
            # to ensure causal relation is still preserved
            input_sequence = Cx.sequence.pad(input_sequence, (window - 1, 0), constant_value=0)

        conv_values = linear(input_sequence)

        a = C.slice(conv_values, 0, 0, hidden_dim)
        b = C.slice(conv_values, 0, hidden_dim, 2 * hidden_dim)
        return a + activation(b)

    return inner


def PositionalEmbedding(max_seq_length: int, hidden_dim: int, init=default_override_or(C.glorot_uniform()),
                        weights=None, name: str = ''):
    """ Learnable positional embedding

    Example:
        a = C.sequence.input_variable(5)
        positional_embedding =

    Arguments:
        max_seq_length (int): max sequence length embeddable
        hidden_dim (int): dimension of the embedding vector
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): (learnable embedding only) initial value of weights `E`
        weights (NumPy array, mutually exclusive with ``init``, defuats to `None`): (user-supplied embedding only) the lookup table.
          The matrix rows are the embedding vectors, ``weights[i,:]`` being the embedding that corresponds to input category `i`.
        name (str): name of the layer

    Returns:
        :class:`~cntk.ops.functions.Function`:
        Positional embedding vector of shape (`hidden_dim`, )
    """

    position_embeddings = Embedding(shape=hidden_dim, init=init, weights=weights, name='PE')

    @C.BlockFunction('PositionalEmbedding', name)
    def inner(x):
        position_index = Cx.sequence.position(x)
        pos = C.one_hot(position_index, max_seq_length, sparse_output=True)
        embedded = position_embeddings(pos)
        return embedded

    return inner


def BertEmbeddings(max_seq_length, hidden_dim: int = None, dropout_rate: float = None,
                   word_embed_init=default_override_or(C.glorot_uniform()), word_embed_weights=None,
                   position_embed_init=default_override_or(C.glorot_uniform()), position_embed_weights=None,
                   token_type_embed_init=default_override_or(C.glorot_uniform()), token_type_embed_weights=None,
                   layer_norm_init_scale=1, layer_norm_init_bias=0, name=''):
    """ Construct the embeddings from word, position and token_type embeddings that is used in BERT.
    Paper can be found at https://arxiv.org/abs/1810.04805 (BERT: Pre-training of Deep Bidirectional
    Transformers for Language Understanding)

    Arguments:
        max_seq_length (int): max sequence length possible for positional embedding
        hidden_dim (int): dimension of the embedding vector
        dropout_rate (float): probability of dropout
        layer_norm_init_scale (float): initial value for the ``scale`` parameter
        layer_norm_init_bias (float): initial value for the ``bias`` parameter

    Returns:
        :class:`~cntk.ops.functions.Function`:
        Embedding vector of shape (`hidden_dim`, )

    """
    word_embeddings = Embedding(shape=hidden_dim, init=word_embed_init, weights=word_embed_weights, name='word_embeddings')
    position_embeddings = PositionalEmbedding(max_seq_length, hidden_dim=hidden_dim, init=position_embed_init, weights=position_embed_weights, name='position_embeddings')
    token_type_embeddings = Embedding(shape=hidden_dim, init=token_type_embed_init, weights=token_type_embed_weights, name='token_type_embeddings')  # aka 'segment embedding'

    layer_norm = LayerNormalization(initial_scale=layer_norm_init_scale, initial_bias=layer_norm_init_bias,
                                    name='LayerNorm')

    dropout = Dropout(dropout_rate, name='dropout')

    @C.BlockFunction('BertEmbeddings', name)
    def inner(text_tensor, token_type_tensor):
        embedded_word_tensors = word_embeddings(text_tensor)
        embedded_token_type_tensors = token_type_embeddings(token_type_tensor)
        embedded_position_tensors = position_embeddings(text_tensor)

        embedded_tensor = embedded_word_tensors + embedded_position_tensors + embedded_token_type_tensors
        embedded_tensor = layer_norm(embedded_tensor)
        embedded_tensor = dropout(embedded_tensor)
        return embedded_tensor

    return inner


def PreTrainedBertEmbeddings(tf_bert_model_filepath: str, dropout_rate: float = None, name=''):
    """ Use pre-trained tensorflow bert model to initialise the model

    Currently it is tested to work with:
        - `BERT-Base, Uncased`, uncased_L-12_H-768_A-12

    Models can be downloaded at https://github.com/google-research/bert

    Arguments:
        tf_bert_model_filepath (str): file path to the tensorflow model
        dropout_rate (float): probability of dropping out an element
        learnable (bool): True if training of embeddings is desired. Defaults to False.

    Returns:
        :class:`~cntk.ops.functions.Function`:
        TF to CNTK Pre-trained Bert Embeddings vector
    """

    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("Loading a TensorFlow models in CNTK, requires TensorFlow to be installed. Please see "
                          "https://www.tensorflow.org/install/ for installation instructions.")

    bert_embedding = 'bert/embeddings/'
    layer_names = [f'{bert_embedding}LayerNorm/beta',
                   f'{bert_embedding}LayerNorm/gamma',
                   f'{bert_embedding}position_embeddings',
                   f'{bert_embedding}token_type_embeddings',
                   f'{bert_embedding}word_embeddings']

    variables_meta = [meta for meta in tf.train.list_variables(tf_bert_model_filepath) if meta[0] in layer_names]
    pretrained_weights = [tf.train.load_variable(tf_bert_model_filepath, meta[0]) for meta in variables_meta]
    pretrained_variables = [(n, shape, weight) for weight, (n, shape) in zip(pretrained_weights, variables_meta)]

    layernorm_beta_embed_variables = pretrained_variables[0]  # bias
    layernorm_gamma_embed_variables = pretrained_variables[1]  # scale
    position_embed_variables = pretrained_variables[2]
    token_type_embed_variables = pretrained_variables[3]
    word_embed_variables = pretrained_variables[4]

    pretrained_bert_embedding = BertEmbeddings(max_seq_length=position_embed_variables[1][0],
                                               hidden_dim=1,  # this argument must be declared and will be ignored
                                               dropout_rate=dropout_rate,
                                               word_embed_init=word_embed_variables[-1],
                                               position_embed_init=position_embed_variables[-1],
                                               token_type_embed_init=token_type_embed_variables[-1],
                                               layer_norm_init_scale=layernorm_gamma_embed_variables[-1],
                                               layer_norm_init_bias=layernorm_beta_embed_variables[-1],
                                               name=name)

    return pretrained_bert_embedding


def BertPooler(shape, init=default_override_or(C.glorot_uniform()), init_bias=default_override_or(0), name=''):
    """ Bert Pooler layer

    We "pool" the model by simply taking the hidden state corresponding to the first token.

    Arguments:
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        name (str, defaults to ''): the name of the function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    dense = Dense(shape=shape, activation=C.tanh, init=init, init_bias=init_bias)

    @C.BlockFunction('BertPooler', name)
    def inner(x):

        return dense(C.sequence.first(x))

    return inner


def PretrainedBertPooler(tf_bert_model_filepath: str):
    """ Pre-trained bert pooler converted from the tensorflow model


    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("Loading a TensorFlow models in CNTK, requires TensorFlow to be installed. Please see "
                          "https://www.tensorflow.org/install/ for installation instructions.")

    pretrained_bert_pooler = BertPooler((None, ),  # shape is not necessary when init from np array
                                        init=tf.train.load_variable(tf_bert_model_filepath, "bert/pooler/dense/kernel"),
                                        init_bias=tf.train.load_variable(tf_bert_model_filepath, "bert/pooler/dense/bias"),
                                        name='pooler')

    return pretrained_bert_pooler


def PositionwiseFeedForward(model_dim: int, intermediate_dim: int, dropout_rate: float = None,
                            intermediate_init=default_override_or(C.glorot_uniform()),  intermediate_init_bias=default_override_or(0),
                            init=default_override_or(C.glorot_uniform()),  init_bias=default_override_or(0),
                            name: str = ''):
    """ Implements Position-wise Feed-Forward Network found in Transformer and BERT

    For more details please refer to "Attention is all you need", https://arxiv.org/abs/1706.03762

    Arguments:
        model_dim (int): dimensionality of model (output)
        intermediate_dim (int): hidden/ intermediate dimension within layer
        dropout_rate (float): probability of dropping out an element
        intermediate_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        intermediate_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the operation to it

    """
    inner_dense = Dense(intermediate_dim, init=intermediate_init, init_bias=intermediate_init_bias, name='intermediate')
    outer_dense = Dense(model_dim, init=init, init_bias=init_bias, name='dense')
    dropout = Dropout(dropout_rate)

    @C.BlockFunction('PositionwiseFeedForward', name)
    def inner(x):
        return outer_dense(dropout(C.relu(inner_dense(x))))

    return inner


def vFSMN(shape, activation, num_past_context, num_future_context, input_rank=None, init=C.glorot_normal(), bias=True,
          init_bias=0, name=''):
    """ Bi-directional vectorised Feedforward sequential memory network

    Implementation of feedforward sequential memory networks (FSMN), to model
    long-term dependency in time series without using recurrent feedback.

    FSMN is a standard fully-connected feedforward neural network equipped
    with some learnable memory blocks in its hidden layers. The memory blocks
    use a tapped-delay line structure to encode the long context information into
    a fixed-size representation as short-term memory mechanism.

    The authors claim that FSMNs can be learned much more reliably and faster than
    RNNs or LSTMs due to the inherent non-recurrent model structure while significantly
    outperforming RNNs in language and speech modeling.

    For more details please refer to "Feedforward Sequential Memory Networks: A New
    Structure to Learn Long-term Dependency" by Zhang, et al.

    Example:
        a = C.sequence.input_variable(10)
        b = vFSMN(100, C.relu, num_past_context=3, num_future_context=0)(a)

        assert b.shape == (100,)

        # bidirectional vFSMN (enable both past and future context)
        a = C.sequence.input_variable(10)
        b = vFSMN(120, C.relu, num_past_context=3, num_future_context=3)(a)

        assert b.shape == (120,)

    Arguments:
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        activation (:class:`~cntk.ops.functions.Function`, defaults to identity): optional function to apply at the end, e.g. `relu`
        num_past_context (int): number of previous frames/ time steps to use to build memory
        num_future_context (int): number of future frames/ time steps to use to build memory
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        input_rank (int, defaults to `None`): number of inferred axes to add to W (`map_rank` must not be given)
        bias (bool, optional, defaults to `True`): the layer will have no bias if `False` is passed here
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the operation to it
    """
    output_shape = _as_tuple(shape)
    output_rank = len(output_shape)   # support outputs with tensor layouts

    if output_rank > 1:
        raise ValueError(f"Shape {output_shape} cannot be 2 dimensional and above")

    # parameters bound to this Function
    if isinstance(init, np.ndarray):
        init_weights = init
    else:
        init_weights = _initializer_for(init, Record(output_rank=output_rank))

    # If input_rank not given then pass a single _INFERRED; map_rank if given will determine the input_rank.
    # The dimension inference may still create multiple axes.
    input_shape = _INFERRED * (input_rank if input_rank is not None else 1)

    W = C.Parameter(shape=input_shape + output_shape, init=init_weights, name='W')
    H = C.Parameter(shape=input_shape + output_shape, init=init_weights, name='H')
    a = C.Parameter(shape=input_shape + _INFERRED, name='a')  # shape = (-1, -1)
    b = C.Parameter(shape=output_shape, init=init_bias,    name='b') if bias else None

    @C.BlockFunction('vFSMN', name)
    def inner(x):
        past = [C.sequence.delay(x, time_step=i + 1) for i in range(num_past_context)]
        future = [C.sequence.delay(x, time_step=i + 1) for i in range(num_future_context)]

        # compute memory
        hidden_memory = C.splice(x, *past, *future, axis=C.Axis.new_leading_axis())
        hidden_memory = a * hidden_memory
        hidden_memory = C.squeeze(C.reduce_sum(hidden_memory, axis=0), axes=0)  # BUGBUG: keepdim must be True

        r = C.times(x, W) + C.times(hidden_memory, H)  # TODO: potential speed up by splice into one big matrix to times

        if bias:
            r = r + b

        if activation is not None:
            r = activation(r)

        return r

    return inner


def cFSMN(shape, proj_dim, activation, num_past_context, num_future_context, input_rank=None, init=C.glorot_normal(), bias=True,
          init_bias=0, name=''):
    """ Bi-directional Compact Feedforward Sequential Memory Network

    cFSMN is a compact version of FSMN that can result in a reduction of up
    to 60% in model size and speed up the learning by more than 7 times while
    still significantly outperforming the popular bi-direction LSTMs for both
    frame-level cross-entropy (CE) criterion based training and MMI based sequence training.

    For more details please refer to "Compact Feedforward Sequential Memory Networks for
    Large VocabularyContinuous Speech Recognition" by Zhang, et al.

    Example:
        a = C.sequence.input_variable(100)
        b = cFSMN(100, 30, C.relu, num_past_context=10, num_future_context=10)(a)

        assert b.shape == (100,)

    Arguments:
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        proj_dim (int): low rank projection of the hidden memory block
        activation (:class:`~cntk.ops.functions.Function`, defaults to identity): optional function to apply at the end, e.g. `relu`
        num_past_context (int): number of previous frames/ time steps to use to build memory
        num_future_context (int): number of future frames/ time steps to use to build memory
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        input_rank (int, defaults to `None`): number of inferred axes to add to W (`map_rank` must not be given)
        bias (bool, optional, defaults to `True`): the layer will have no bias if `False` is passed here
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the operation to it
    """
    output_shape = _as_tuple(shape)
    output_rank = len(output_shape)   # support outputs with tensor layouts

    if output_rank > 1:
        raise ValueError(f"Shape {output_shape} cannot be 2 dimensional and above")

    # parameters bound to this Function
    if isinstance(init, np.ndarray):
        init_weights = init
    else:
        init_weights = _initializer_for(init, Record(output_rank=output_rank))

    # If input_rank not given then pass a single _INFERRED; map_rank if given will determine the input_rank.
    # The dimension inference may still create multiple axes.
    input_shape = _INFERRED * (input_rank if input_rank is not None else 1)

    linear = Dense(shape=proj_dim, init=init, bias=bias, init_bias=init_bias, input_rank=input_rank, name='projection')
    H = C.Parameter(shape=input_shape + output_shape, init=init_weights, name='H')
    a = C.Parameter(shape=input_shape + _INFERRED, name='a')  # shape = (-1, -1)  # TODO: input_shape input_rank
    b = C.Parameter(shape=output_shape, init=init_bias, name='bb') if bias else None

    @C.BlockFunction('cFSMN', name)
    def inner(x):
        p = linear(x)
        past = [C.sequence.delay(p, time_step=i + 1) for i in range(num_past_context)]
        future = [C.sequence.delay(p, time_step=i + 1) for i in range(num_future_context)]

        # compute memory
        hidden_memory = C.splice(p, *past, *future, axis=C.Axis.new_leading_axis())
        hidden_memory = a * hidden_memory
        hidden_memory = p + C.squeeze(C.reduce_sum(hidden_memory, axis=0), axes=0)  # BUGBUG: keepdim must be True

        r = C.times(hidden_memory, H)

        if bias:
            r = r + b

        if activation is not None:
            r = activation(r)

        return r

    return inner
