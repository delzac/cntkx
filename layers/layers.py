import math
import numpy as np
import cntk as C
import cntkx as Cx
from cntkx.layers.blocks import WeightMaskedLSTM
from cntk.default_options import default_override_or
from cntk.layers.blocks import identity
from cntk.layers import SequentialConvolution, Recurrence, Embedding, LayerNormalization, Dropout
from cntk.layers import MaxPooling, Convolution2D


def QRNN(window: int = 1, hidden_dim=None, activation=C.tanh, return_full_state=False):
    """
    Quasi-Recurrent Neural Networks layer

    This is the CNTK implementation of [Salesforce Research](https://einstein.ai/)'s
    [Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576) paper.

    More details on tuning and application can be found in this paper:
    [An Analysis of Neural Language Modeling at Multiple Scales](https://arxiv.org/abs/1803.08240)

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

    Returns:
        :class:`~cntk.ops.functions.Function`: OR
        tuple of :class:`~cntk.ops.functions.Function`:

    """

    @C.Function
    def f_pool(c, zf):
        z = C.slice(zf, 0, 0, hidden_dim)
        f = C.slice(zf, 0, hidden_dim, 2 * hidden_dim)
        return f * c + (1 - f) * z

    def model(input_tensor):
        filter_shape = (window, ) + input_tensor.shape

        input_sequence = input_tensor
        if window > 1:
            # to ensure causal relation is still preserved
            input_sequence = Cx.sequence.pad(input_sequence, (window - 1, 0), constant_value=0)

        gate_values = SequentialConvolution(filter_shape=filter_shape, num_filters=3 * hidden_dim, pad=False,
                                            reduction_rank=0)(input_sequence) >> C.squeeze

        x = C.slice(gate_values, -1, 0, hidden_dim)
        forget = C.slice(gate_values, -1, hidden_dim, 2 * hidden_dim)
        output = C.slice(gate_values, -1, 2 * hidden_dim, 3 * hidden_dim)

        z = activation(x)
        f = C.sigmoid(forget)
        o = C.sigmoid(output)

        # Pooling
        c = Recurrence(f_pool)(C.splice(z, f))  # f pool
        h = o * c  # o pool

        if return_full_state:
            return h, c
        else:
            return h

    return model


def SinusoidalPositionalEmbedding(min_timescale=1.0, max_timescale=1.0e4, name: str = ''):
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
        b = Cx.layers.SinusoidalPositionalEmbedding()(a)

        assert b.shape == (10, )

    Arguments:
        min_timescale (float): geometric sequence of timescales starting with min_timescale
        max_timescale (float): geometric sequence of timescales ending with max_timescale
        name (str): a name for this layer.

    Returns:
        :class:`~cntk.ops.functions.Function`: same shape as input sequence tensor

    """

    @C.Function
    def position(p, x):
        return p + x * 0 + 1

    def embedding(x):
        assert x.shape[0] > 0, f"input tensor must have a defined shape, input shape is {x.shape}"
        dim = x.shape[0]
        num_timescales = dim // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1))
        inv_timescales = C.constant(min_timescale * np.exp(np.arange(num_timescales) * -log_timescale_increment),
                                    dtype=np.float32)

        pos = Recurrence(position)(C.slice(x, 0, 0, num_timescales))
        scaled_time = pos * inv_timescales
        s = C.sin(scaled_time)
        c = C.cos(scaled_time)
        signal = C.splice(s, c)

        # last dim gets a 0 value if input_dim is odd
        if dim % 2 != 0:
            signal = C.pad(signal, [[0, 1]])

        return C.layers.Label(name=name)(signal)

    return embedding


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
                  pool_strides=1,
                  pool_pad=default_override_or(False),
                  name_prefix=''):
    """ Stack of Convolution 2D followed by one max pooling layer. Convenience wrapper. """

    conv_stack = Convolution2DStack(n, conv_filter_shape, conv_num_filters, activation, init, conv_pad, conv_strides,
                                    bias, init_bias, reduction_rank, dilation, groups, name_prefix)

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
                       name_prefix=''):
    """ A stack of of convolutional layers. Convenience wrapper. """

    convs = [Convolution2D(filter_shape, num_filters, activation, init, pad, strides, bias,
                           init_bias, reduction_rank, dilation, groups,
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


def GatedLinearUnit(window: int = 2, hidden_dim: int = None, activation=C.sigmoid):
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
        activation: gate function

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    assert hidden_dim % 2 == 0, "hidden dimension must be divisible by 2"

    def inner(input_tensor):
        filter_shape = (window,) + input_tensor.shape

        input_sequence = input_tensor
        if window > 1:
            # to ensure causal relation is still preserved
            input_sequence = Cx.sequence.pad(input_sequence, (window - 1, 0), constant_value=0)

        conv_values = SequentialConvolution(filter_shape=filter_shape, num_filters=2 * hidden_dim, pad=False,
                                            reduction_rank=0)(input_sequence) >> C.squeeze

        outputs = C.slice(conv_values, 0, 0, hidden_dim) + activation(C.slice(conv_values, 0, hidden_dim, 2 * hidden_dim))
        return outputs

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

    position_embeddings = Embedding(shape=hidden_dim, init=init, weights=weights, name=name)

    @C.BlockFunction('PositionalEmbedding', name)
    def inner(x):
        position_index = Cx.sequence.position(x)
        pos = C.one_hot(position_index, max_seq_length, sparse_output=True) >> C.squeeze
        embedded = position_embeddings(pos)
        return embedded

    return inner


def BertEmbeddings(max_seq_length, hidden_dim: int = None, dropout_rate: float = None,
                   word_embed_init=default_override_or(C.glorot_uniform()), word_embed_weights=None,
                   position_embed_init=default_override_or(C.glorot_uniform()), position_embed_weights=None,
                   token_type_embed_init=default_override_or(C.glorot_uniform()), token_type_embed_weights=None,
                   layer_norm_init_scale=1, layer_norm_init_bias=0,
                   name=''):
    """ Construct the embeddings from word, position and token_type embeddings that is used in BERT.
    Paper can be found at https://arxiv.org/abs/1810.04805 (BERT: Pre-training of Deep Bidirectional
    Transformers for Language Understanding)

    Arguments:
        max_seq_length (int): max sequence length possible for positional embedding
        hidden_dim (int): dimension of the embedding vector
        dropout_rate (float): probability of dropout
        layer_norm_init_scale (float): initial value for the ``scale`` parameter
        layer_norm_init_bias (float): initial value for the ``bias`` parameter
        pretrianed_bert (str): file path to bert pretrained model file

    Returns:
        :class:`~cntk.ops.functions.Function`:
        Embedding vector of shape (`hidden_dim`, )

    """
    word_embeddings = Embedding(shape=hidden_dim, init=word_embed_init, weights=word_embed_weights, name='word_embeddings')
    position_embeddings = PositionalEmbedding(max_seq_length, hidden_dim=hidden_dim, init=position_embed_init, weights=position_embed_weights, name='position_embeddings')
    token_type_embeddings = Embedding(shape=hidden_dim, init=token_type_embed_init, weights=token_type_embed_weights, name='token_type_embeddings')  # aka 'segment embedding'

    layer_norm = LayerNormalization(initial_scale=layer_norm_init_scale, initial_bias=layer_norm_init_bias, name='LayerNorm')
    dropout = Dropout(dropout_rate)

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


def PreTrainedBertEmbeddings(tf_bert_model_filepath: str, dropout_rate: float = None, learnable: bool = False):
    """ Use pre-trained tensorflow bert model to initialise the model

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
    pretrained_variables = [(name, shape, weight) for weight, (name, shape) in zip(pretrained_weights, variables_meta)]

    layernorm_beta_embed_variables = pretrained_variables[0]  # bias
    layernorm_gamma_embed_variables = pretrained_variables[1]  # scale
    position_embed_variables = pretrained_variables[2]
    token_type_embed_variables = pretrained_variables[3]
    word_embed_variables = pretrained_variables[4]

    if not learnable:
        pretrained_bert_embedding = BertEmbeddings(max_seq_length=position_embed_variables[1][0],
                                                   hidden_dim=None,
                                                   dropout_rate=dropout_rate,
                                                   word_embed_weights=word_embed_variables[-1],
                                                   position_embed_weights=position_embed_variables[-1],
                                                   token_type_embed_weights=token_type_embed_variables[-1],
                                                   layer_norm_init_scale=layernorm_gamma_embed_variables[-1],
                                                   layer_norm_init_bias=layernorm_beta_embed_variables[-1])
    else:
        pretrained_bert_embedding = BertEmbeddings(max_seq_length=position_embed_variables[1][0],
                                                   hidden_dim=None,
                                                   dropout_rate=dropout_rate,
                                                   word_embed_init=word_embed_variables[-1],
                                                   position_embed_init=position_embed_variables[-1],
                                                   token_type_embed_init=token_type_embed_variables[-1],
                                                   layer_norm_init_scale=layernorm_gamma_embed_variables[-1],
                                                   layer_norm_init_bias=layernorm_beta_embed_variables[-1])

    return pretrained_bert_embedding


def WeightDroppedLSTM(shape, dropconnect_rate: float = None, variational_dropout_rate_input: float = None,
                      variational_dropout_rate_output: float = None,
                      activation=default_override_or(C.tanh), use_peepholes=default_override_or(False),
                      init=default_override_or(C.glorot_uniform()), init_bias=default_override_or(0),
                      enable_self_stabilization=default_override_or(False), go_backwards=default_override_or(False),
                      initial_state=default_override_or(0), return_full_state=False, name=''):
    """ LSTM recurence layer with DropConnect and variational dropout applied

    Weight dropped is implemented as DropConnect of hidden-to-hidden weight matrics, not the dropout of
    hidden states (aka variational dropout).

    For more details on Weight-Dropped LSTM, please read "regularizing and optimizing LSTM language models"
    by S. Merity, at el (https://arxiv.org/abs/1708.02182)

    Weight masked LSTM step function is available in cntkx.layers.blocks as WeightMaskedLSTM.

    Note that in typical usage, the output of the last `WeightDroppedLSTM` layer in the rnn layer stack
    should not be variationally dropped out (i.e. variational_dropout_rate_output should be set to zero).
    This is advice is consistent with salesforce's implementation of awd-lstm
    (https://github.com/salesforce/awd-lstm-lm/blob/master/model.py)

    Examples:
        a = C.sequence.input_variable(10)
        b = Cx.layers.WeightDroppedLSTM(20, 0.1, 0.1, 0.1)(a)

        assert b.shape == (20, )

    Arguments:
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        dropconnect_rate (float): probability of dropping out an element in dropconnect
        variational_dropout_rate_input (float): probability of dropping out an input element
        variational_dropout_rate_output (float): probability of dropping out an output element
        activation (:class:`~cntk.ops.functions.Function`, defaults to :func:`~cntk.ops.tanh`): function to apply at the end, e.g. `relu`
        use_peepholes (bool, defaults to `False`):
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to `glorot_uniform`): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        enable_self_stabilization (bool, defaults to `False`): if `True` then add a :func:`~cntk.layers.blocks.Stabilizer`
         to all state-related projections (but not the data input)
        go_backwards (bool, defaults to ``False``): if ``True`` then run the recurrence from the end of the sequence to the start.
        initial_state (scalar or tensor without batch dimension; or a tuple thereof):
          the initial value for the state. This can be a constant or a learnable parameter.
          In the latter case, if the step function has more than 1 state variable,
          this parameter must be a tuple providing one initial state for every state variable.
        return_full_state (bool, defaults to ``False``): if ``True`` and the step function has more than one
          state variable, then the layer returns a all state variables (a tuple of sequences);
          whereas if not given or ``False``, only the first state variable is returned to the caller.
        name (str, defaults to ''): the name of the Function instance in the network

    """
    dropout = C.layers.Dropout(dropconnect_rate, name='dropout')
    variational_dropout_input = VariationalDropout(variational_dropout_rate_input) if variational_dropout_rate_input > 0 else None
    variational_dropout_output = VariationalDropout(variational_dropout_rate_output) if variational_dropout_rate_output > 0 else None

    lstm = WeightMaskedLSTM(shape=shape, activation=activation, use_peepholes=use_peepholes, init=init, init_bias=init_bias,
                            enable_self_stabilization=enable_self_stabilization, name=name + '_WeightMaskedLSTM_cell')

    @C.Function
    def inner(x):

        # mask for hidden-to-hidden weight that is the same for all temporal steps
        dummy = C.slice(C.sequence.first(x), 0, 0, 1)
        drop_connect = dropout(C.zeros_like(lstm.parameters[-1]) * dummy + C.constant(1))
        drop_connect = C.sequence.broadcast_as(drop_connect, x)

        @C.Function
        def weight_dropped_lstm(h, c, x):

            a, b, __ = lstm(h, c, drop_connect, x).outputs
            return a, b

        x = variational_dropout_input(x) if variational_dropout_input else x
        output = Recurrence(weight_dropped_lstm, go_backwards=go_backwards, initial_state=initial_state,
                            return_full_state=return_full_state, name=name)(x)

        # dropout applied outside of rnn as rnn hidden-to-hidden already regularised by dropconnect
        output = variational_dropout_output(output) if variational_dropout_output else output
        return output

    return inner


def VariationalDropout(dropout_rate: float, name=''):
    """ Variational dropout uses the same dropout mask at each time step (i.e. across the dynamic sequence axis)

    Example:
        a = C.sequence.input_variable(10)
        b = VariationalDropout(0.1)(a)

        assert b.shape == a.shape

    Arguments:
        dropout_rate (float): probability of dropping out an element
        name (str, defaults to ''): the name of the Function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the operation to it

    """
    dropout = C.layers.Dropout(dropout_rate, name=name)

    @C.BlockFunction('VariationalDropout', name)
    def inner(x):
        mask = dropout(C.sequence.first(x))
        mask = C.sequence.broadcast_as(mask, x)
        return mask * x

    return inner
