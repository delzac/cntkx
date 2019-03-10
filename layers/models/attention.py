import cntk as C
import cntkx as Cx
import numpy as np
from cntk.layers import LayerNormalization, ResNetBlock, Recurrence
from cntkx.layers import PreTrainedBertEmbeddings, PositionwiseFeedForward, Dense, PretrainedBertPooler
from cntk.default_options import default_override_or
from cntk.layers.blocks import _inject_name


def ScaledDotProductAttention(obey_sequence_order: bool = None, max_seq_len: int = None, name=''):
    """
    Scaled dot-product attention implementation of "Attention is all you need", https://arxiv.org/abs/1706.03762

    An attention function can be described as mapping a query and a set of key-value pairs to an output,
    where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
    of the values, where the weight assigned to each value is computed by a compatibility function of the
    query with the corresponding key.

    scaled_dot_product_attention(Q, K, V) = softmax(QV.T / sqrt(dk)) * V

    When query, key and value are all the same, it becomes self-attention.

    Note:
        Query and key must have the same dimension
        Key and value must have the same sequence length

    Example:
        a = C.sequence.input_variable(10)
        b = ScaledDotProductAttention()(a, a, a)

        assert b.shape == (10, )

        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed

    Returns:
        :class:`~cntk.ops.functions.Function`:
        A function that returns a weighted sum of value

    """

    @C.BlockFunction('ScaledDotProductAttention', name)
    def attention(query, key, value):
        dk = C.reduce_sum(C.ones_like(query))  # cannot use sequence.last, will conflict with recurrence
        # dk: [#, *] [1, ] and value = int(dim_of_query)

        unpacked_key = C.sequence.unpack(key, padding_value=0, no_mask_output=True)  # [#] [-3, key_dim]
        unpacked_value = C.sequence.unpack(value, padding_value=0, no_mask_output=True)  # [#] [-3, value_dim]

        broadcasted_key = C.sequence.broadcast_as(unpacked_key, query)  # [#, *] [-3, key_dim]
        scaled = C.times_transpose(query, broadcasted_key) / dk
        # [#, *] [q_dim] @ [#, *] [key_dim, -3], assert q_dim == key_dim
        # scaled: [#, *] [-3, ] => for every key seq element, there is a corresponding score

        # masked out invalid temporal connections to obey_sequence_order
        if obey_sequence_order and max_seq_len:
            unpacked_scaled, scaled_mask = C.sequence.unpack(scaled, padding_value=0).outputs
            # unpacked_scaled: [#] [-3, -3]  <== matrix will be top right diagonally zero-ed
            # scaled_mask: [#] [-3,]

            minus_inf = C.constant(-1e+30)
            valid_connections = C.Constant(np.tril(np.ones((max_seq_len, max_seq_len)), k=0))  # [] [max_seq, max_seq]
            valid_connections = C.reconcile_dynamic_axes(valid_connections, unpacked_scaled)  # [#] [max_seq, max_seq]
            valid_connections = C.crop_manual(valid_connections, unpacked_scaled, 0, 0)  # [#] [-3, -3]
            unpacked_scaled = C.element_select(valid_connections, unpacked_scaled, minus_inf)  # [#] [-3, -3]
            scaled = C.to_sequence_like(unpacked_scaled, query)  # [#, *] [-3]

        elif obey_sequence_order and not max_seq_len:
            raise ValueError("max_seq_len must be defined when obey_sequence_order is True")

        attended = C.times(C.softmax(scaled, axis=-1), C.sequence.broadcast_as(unpacked_value, query))  # [#, *] [value_dim,]
        return attended

    return attention


def MultiHeadAttention(num_heads, model_dim, obey_sequence_order: bool = None, max_seq_len: int = None,
                       key_init=default_override_or(C.glorot_uniform()), key_init_bias=default_override_or(0),
                       query_init=default_override_or(C.glorot_uniform()), query_init_bias=default_override_or(0),
                       value_init=default_override_or(C.glorot_uniform()), value_init_bias=default_override_or(0),
                       init=default_override_or(C.glorot_uniform()), init_bias=default_override_or(0),
                       name=''):
    """ Multi-head attention as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Example:
        a = C.sequence.input_variable(10)
        b = MultiHeadAttention(2, 10)(a, a, a)

        assert b.shape == (10, )

    Arguments:
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed
        key_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        key_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        query_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        query_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        value_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        value_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    assert model_dim % num_heads == 0, "Model dimension must be divisible by number of heads"

    head_dim = int(model_dim / num_heads)

    query_linear = Dense(model_dim, init=query_init, init_bias=query_init_bias)
    key_linear = Dense(model_dim, init=key_init, init_bias=key_init_bias)
    value_linear = Dense(model_dim, init=value_init, init_bias=value_init_bias)
    multihead_liner = Dense(model_dim, init=init, init_bias=init_bias)

    scaled_dot_product_attention = ScaledDotProductAttention(obey_sequence_order, max_seq_len)

    @C.BlockFunction('MultiHeadAttention', name)
    def inner(query, key, value):
        mixed_queries = query_linear(query)  # [#, *] {model_dim,]
        mixed_keys = key_linear(key)  # [#, *] {model_dim,]
        mixed_values = value_linear(value)  # [#, *] {model_dim,]

        # TODO: re-implement `ScaledDotProductAttention` when cntk has BatchMatMul so there's no need to slice here
        queries = [C.slice(mixed_queries, 0, i * head_dim, (i + 1) * head_dim) for i in range(num_heads)]
        keys = [C.slice(mixed_keys, 0, i * head_dim, (i + 1) * head_dim) for i in range(num_heads)]
        values = [C.slice(mixed_values, 0, i * head_dim, (i + 1) * head_dim) for i in range(num_heads)]

        # list of num_heads heads with shape (-3, head_dim) each
        attention_outputs = [scaled_dot_product_attention(q, k, v) for q, k, v in zip(queries, keys, values)]

        result = multihead_liner(C.splice(*attention_outputs))
        return result

    return _inject_name(inner, name)


def MultiHeadAttentionBlock(num_heads, model_dim, obey_sequence_order: bool = None, max_seq_len: int = None,
                            key_init=default_override_or(C.glorot_uniform()), key_init_bias=default_override_or(0),
                            query_init=default_override_or(C.glorot_uniform()), query_init_bias=default_override_or(0),
                            value_init=default_override_or(C.glorot_uniform()), value_init_bias=default_override_or(0),
                            init=default_override_or(C.glorot_uniform()), init_bias=default_override_or(0),
                            initial_scale=1, initial_bias=0, name=''):
    """ Multi head attention block as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Multi-head attention block comes with a residual connection and a layer norm.

    Example:
        a = C.sequence.input_variable(10)
        b = MultiHeadAttentionBlock(2, 10)(a, a, a)

        assert b.shape == (10, )

    Arguments:
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed
        key_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        key_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        query_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        query_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        value_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        value_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        initial_scale (float, default 1): initial value for the ``scale`` parameter aka gamma
        initial_bias (float, default 0): initial value for the ``bias`` parameter aka beta

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    attention_layer = MultiHeadAttention(num_heads, model_dim, obey_sequence_order, max_seq_len,
                                         key_init=key_init, key_init_bias=key_init_bias,
                                         query_init=query_init, query_init_bias=query_init_bias,
                                         value_init=value_init, value_init_bias=value_init_bias,
                                         init=init, init_bias=init_bias, name='MultiheadAttention')

    layernorm = LayerNormalization(initial_scale=initial_scale, initial_bias=initial_bias, name='LayerNorm')

    @C.Function
    def inner(query, key, value):
        attended = attention_layer(query, key, value)
        skip_connect_attended = attended + query
        normed_skip_connect_attended = layernorm(skip_connect_attended)
        return normed_skip_connect_attended

    return _inject_name(inner, name)


def TransformerEncoderBlock(num_heads: int, model_dim: int, intermediate_dim: int, dropout_rate: float = None,
                            obey_sequence_order: bool = None, max_seq_len: int = None,
                            key_init=default_override_or(C.glorot_uniform()), key_init_bias=default_override_or(0),
                            query_init=default_override_or(C.glorot_uniform()), query_init_bias=default_override_or(0),
                            value_init=default_override_or(C.glorot_uniform()), value_init_bias=default_override_or(0),
                            mha_init=default_override_or(C.glorot_uniform()), mha_init_bias=default_override_or(0),
                            mha_initial_scale=1, mha_initial_bias=0,
                            intermediate_init=default_override_or(C.glorot_uniform()), intermediate_init_bias=default_override_or(0),
                            init=default_override_or(C.glorot_uniform()), init_bias=default_override_or(0),
                            initial_scale=1, initial_bias=0, name=''):
    """ Encoder block of transformer as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Consist of 1 multi head attention followed by a dense layer, residual connect and layer norm

    Arguments:
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        intermediate_dim (int): hidden/ intermediate dimension within position-wise feed-forward layer
        dropout_rate (float): probability of dropping out an element in the position-wise feed-forward
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed
        key_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        key_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        query_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        query_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        value_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        value_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        mha_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        mha_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
         mha_initial_scale (float, default 1): initial value for the ``scale`` parameter aka gamma
        mha_initial_bias (float, default 0): initial value for the ``bias`` parameter aka beta
        intermediate_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        intermediate_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        initial_scale (float, default 1): initial value for the ``scale`` parameter aka gamma
        initial_bias (float, default 0): initial value for the ``bias`` parameter aka beta

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    mha_block = MultiHeadAttentionBlock(num_heads, model_dim, obey_sequence_order, max_seq_len,
                                        key_init=key_init, key_init_bias=key_init_bias,
                                        query_init=query_init, query_init_bias=query_init_bias,
                                        value_init=value_init, value_init_bias=value_init_bias,
                                        init=mha_init, init_bias=mha_init_bias,
                                        initial_scale=mha_initial_scale, initial_bias=mha_initial_bias,
                                        name='SelfAttention')

    feed_foward = PositionwiseFeedForward(model_dim, intermediate_dim, dropout_rate=dropout_rate,
                                          intermediate_init=intermediate_init, intermediate_init_bias=intermediate_init_bias,
                                          init=init, init_bias=init_bias, name='PWFF')

    layernorm = LayerNormalization(initial_scale, initial_bias, name='LayerNorm')

    @C.Function
    def block(x):
        self_attended = mha_block(x, C.alias(x), C.alias(x))
        hidden = feed_foward(self_attended)
        output = layernorm(hidden + self_attended)  # residual connection
        return output

    return _inject_name(block, name)  # consider change to BlockFunction


def TransformerDecoderBlock(num_heads: int, model_dim: int, intermediate_dim: int, dropout_rate: float = None,
                            obey_sequence_order: bool = True, max_seq_len: int = None,
                            mha1_key_init=default_override_or(C.glorot_uniform()), mha1_key_init_bias=default_override_or(0),
                            mha1_query_init=default_override_or(C.glorot_uniform()), mha1_query_init_bias=default_override_or(0),
                            mha1_value_init=default_override_or(C.glorot_uniform()), mha1_value_init_bias=default_override_or(0),
                            mha1_init=default_override_or(C.glorot_uniform()), mha1_init_bias=default_override_or(0),
                            mha1_initial_scale=1, mha1_initial_bias=0,
                            mha2_key_init=default_override_or(C.glorot_uniform()), mha2_key_init_bias=default_override_or(0),
                            mha2_query_init=default_override_or(C.glorot_uniform()), mha2_query_init_bias=default_override_or(0),
                            mha2_value_init=default_override_or(C.glorot_uniform()), mha2_value_init_bias=default_override_or(0),
                            mha2_init=default_override_or(C.glorot_uniform()), mha2_init_bias=default_override_or(0),
                            mha2_initial_scale=1, mha2_initial_bias=0,
                            intermediate_init=default_override_or(C.glorot_uniform()),
                            intermediate_init_bias=default_override_or(0),
                            init=default_override_or(C.glorot_uniform()), init_bias=default_override_or(0),
                            initial_scale=1, initial_bias=0):
    """ Decoder block of transformer as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Consist of 2 multi head attention followed by a dense layer, residual connect and layer norm

    Arguments:
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        intermediate_dim (int): hidden/ intermediate dimension within position-wise feed-forward layer
        dropout_rate (float): probability of dropping out an element in the position-wise feed-forward
        obey_sequence_order (bool, defaults True): do not let attention peek into future values
        max_seq_len (int): max sequence length possible, used to ensure that sequence order is obeyed
        mha1_key_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        mha1_key_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        mha1_query_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        mha1_query_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        mha1_value_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        mha1_value_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        mha1_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        mha1_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        mha1_initial_scale (float, default 1): initial value for the ``scale`` parameter aka gamma
        mha1_initial_bias (float, default 0): initial value for the ``bias`` parameter aka beta
        mha2_key_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        mha2_key_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        mha2_query_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        mha2_query_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        mha2_value_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        mha2_value_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        mha2_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        mha2_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        mha2_initial_scale (float, default 1): initial value for the ``scale`` parameter aka gamma
        mha2_initial_bias (float, default 0): initial value for the ``bias`` parameter aka beta
        intermediate_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        intermediate_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        initial_scale (float, default 1): initial value for the ``scale`` parameter aka gamma
        initial_bias (float, default 0): initial value for the ``bias`` parameter aka beta

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    mha_block1 = MultiHeadAttentionBlock(num_heads=num_heads, model_dim=model_dim,
                                         obey_sequence_order=obey_sequence_order, max_seq_len=max_seq_len,
                                         key_init=mha1_key_init, key_init_bias=mha1_key_init_bias,
                                         query_init=mha1_query_init, query_init_bias=mha1_query_init_bias,
                                         value_init=mha1_value_init, value_init_bias=mha1_value_init_bias,
                                         init=mha1_init, init_bias=mha1_init_bias,
                                         initial_scale=mha1_initial_scale, initial_bias=mha1_initial_bias)
    
    mha_block2 = MultiHeadAttentionBlock(num_heads=num_heads, model_dim=model_dim,
                                         obey_sequence_order=False, max_seq_len=None,
                                         key_init=mha2_key_init, key_init_bias=mha2_key_init_bias,
                                         query_init=mha2_query_init, query_init_bias=mha2_query_init_bias,
                                         value_init=mha2_value_init, value_init_bias=mha2_value_init_bias,
                                         init=mha2_init, init_bias=mha2_init_bias,
                                         initial_scale=mha2_initial_scale, initial_bias=mha2_initial_bias)

    feed_foward = PositionwiseFeedForward(model_dim, intermediate_dim, dropout_rate=dropout_rate,
                                          intermediate_init=intermediate_init, intermediate_init_bias=intermediate_init_bias,
                                          init=init, init_bias=init_bias)

    layernorm = LayerNormalization(initial_scale, initial_bias)

    @C.Function
    def block(encoded, x):
        inner = mha_block1(x, x, x)
        inner = mha_block2(inner, encoded, encoded)
        output = layernorm(ResNetBlock(feed_foward)(inner))
        return output

    return block


def TransformerEncoder(n: int, num_heads: int, model_dim: int, intermediate_dim: int, dropout_rate: float = None):
    """ Transformer encoder as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Example:
        a = C.sequence.input_variable(10)
        encoded = TransformerDecoder(3, 2, 10)(a)

        assert encoded.shape == (10, )

    Arguments:
        n (int): number of encoder blocks
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        intermediate_dim (int): hidden/ intermediate dimension within position-wise feed-forward layer
        dropout_rate (float): probability of dropping out an element in the position-wise feed-forward

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """

    blocks = [TransformerEncoderBlock(num_heads=num_heads, model_dim=model_dim, intermediate_dim=intermediate_dim,
                                      dropout_rate=dropout_rate, obey_sequence_order=False,
                                      max_seq_len=None) for __ in range(n)]

    @C.Function
    def inner(x):

        for block in blocks:
            x = block(x)

        return x

    return inner


def TransformerDecoder(n: int, num_heads: int, model_dim: int, intermediate_dim: int, dropout_rate: float = None,
                       max_seq_len: int = None):
    """ Transformer decoder as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Example:
        a = C.sequence.input_variable(10)
        encoded = C.sequence.input_variable(10)

        decoded = TransformerDecoder(3, 2, 10)(encoded, a)

        assert decoded.shape == (10, )

    Arguments:
        n (int): number of decoder blocks
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        intermediate_dim (int): hidden/ intermediate dimension within position-wise feed-forward layer
        dropout_rate (float): probability of dropping out an element in the position-wise feed-forward
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """

    blocks = [TransformerDecoderBlock(num_heads=num_heads, model_dim=model_dim, intermediate_dim=intermediate_dim,
                                      dropout_rate=dropout_rate, obey_sequence_order=True, max_seq_len=max_seq_len)
              for __ in range(n)]

    @C.Function
    def decoder(encoded, x):

        for block in blocks:
            x = block(encoded, x)

        return x

    return decoder


def Transformer(num_encoder_blocks: int = 6, num_decoder_blocks=6, num_heads_encoder: int = 16,
                num_heads_decoder: int = 16, encoder_model_dim: int = 512, decoder_model_dim: int = 512,
                encoder_intermediate_dim: int = 2048, decoder_intermediate_dim: int = 2048,
                encoder_dropout_rate: float = 0.1, decoder_dropout_rate: float = 0.1,
                max_seq_len_decoder: int = 100):
    """ Transformer implementation as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Example:
        a = C.sequence.input_variable(512)
        b = C.sequence.input_variable(512)

        transformer = Transformer()  # using default settings
        decoded = transformer(a, b)

        assert decoded.shape == (512, )

    Arguments:
        num_encoder_blocks: number of encoder blocks
        num_decoder_blocks: number of decoder blocks
        num_heads_encoder: number of encoder attention heads
        num_heads_decoder: number of decoder attention heads
        encoder_model_dim: encoder model output dimension (should be the same dimension as the transformer input)
        decoder_model_dim: decoder model output dimension (should be the same dimension as the transformer input)
        encoder_intermediate_dim (int): hidden/ intermediate dimension within position-wise feed-forward layer of encoder
        decoder_intermediate_dim (int): hidden/ intermediate dimension within position-wise feed-forward layer of decoder
        encoder_dropout_rate (float): probability of dropping out an element in the position-wise feed-forward of encoder
        decoder_dropout_rate (float): probability of dropping out an element in the position-wise feed-forward of decoder
        max_seq_len_decoder: max sequence length in decoding sequence. Used for preventing attention peeking into future values.

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    encoder = TransformerEncoder(n=num_encoder_blocks, num_heads=num_heads_encoder, model_dim=encoder_model_dim,
                                 intermediate_dim=encoder_intermediate_dim, dropout_rate=encoder_dropout_rate)

    decoder = TransformerDecoder(n=num_decoder_blocks, num_heads=num_heads_decoder, model_dim=decoder_model_dim,
                                 intermediate_dim=decoder_intermediate_dim, dropout_rate=decoder_dropout_rate,
                                 max_seq_len=max_seq_len_decoder)

    @C.Function
    def model(tensor_to_encode, decoder_input_tensor):
        encoded = encoder(tensor_to_encode)
        decoded = decoder(encoded, decoder_input_tensor)
        return decoded

    return model


def GaussianWindowAttention(nb_mixtures):
    """
    Implementation of the attention model found in "Generating sequences with recurrent neural networks" by Alex Graves.

    Gaussian window attention uses a directional mixture of gaussian kernels as convolution/attention window.

    For more details, the paper can be found in https://arxiv.org/abs/1308.0850

    Example:
        seq1 = C.Axis.new_unique_dynamic_axis('seq1')
        seq2 = C.Axis.new_unique_dynamic_axis('seq2')

        encoded = C.sequence.input_variable(30, sequence_axis=seq1)
        query = C.sequence.input_variable(28, sequence_axis=seq2)

        a = GaussianWindowAttention(10)(encoded, query)

        assert a.shape == (30, )

    Arguments:
        nb_mixtures (int): number of gaussian mixtures to use for attention model

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    dense = Dense(shape=3 * nb_mixtures, activation=None, init=C.normal(0.075), name="GravesAttention")

    def window_weight(a, b, k, u):
        """
        Calculate Phi is the window weight of character seq at position u of time t.
        Function tested to be correct on 2018-25-02 using numpy equivalent

        math:
            phi = summation of mixtures { a * exp ( -b * (k - u) ^ 2 ) }

        Args:
            a: importance of window within the mixture. Not normalised and doesn't sum to one.
            b: width of attention window
            k: location of window
            u: integer position of each item in sequence. Value from 1 to seq_length. (rank 2 tensor) [-3, 1]

        Returns:
            :class:`~cntk.ops.functions.Function`

        """
        # print(f"k shape: {k.shape}, u shape: {u.shape}")
        phi = a * C.exp(-1 * b * C.square(k - u))
        # print("internal phi shape:", phi.shape)
        phi = C.swapaxes(C.reduce_sum(phi, axis=0))  # Reduce sum the mixture axis
        # phi: [#, n] [*-c, 1]
        return phi

    @C.typemap
    def gaussian_windows_attention_coefficients(abk, nb_mixtures):
        """ Split into 3 equal tensor of dim nb_mixtures """
        a = C.exp(C.slice(abk, 0, 0, nb_mixtures))
        b = C.exp(C.slice(abk, 0, nb_mixtures, 2 * nb_mixtures))
        k = C.exp(C.slice(abk, 0, 2 * nb_mixtures, 0))
        k = Recurrence(C.plus)(k)

        a = C.expand_dims(a, axis=-1)
        b = C.expand_dims(b, axis=-1)
        k = C.expand_dims(k, axis=-1)
        return a, b, k

    @C.Function
    def attention(encoded, network):
        abk = dense(network)
        a, b, k = gaussian_windows_attention_coefficients(abk, nb_mixtures)
        # print("abk shape:", a.shape, b.shape, k.shape)
        # a, b, k: [#, n] [nb_mixture, 1]
        # context: [#, c] [char_ohe]

        encoded_unpacked = C.sequence.unpack(encoded, padding_value=0, no_mask_output=True)
        # context_unpacked: [#] [*=c, char_ohe]
        u = Cx.sequence.position(encoded)
        # u: [#, c], [1]
        u_values, u_valid = C.sequence.unpack(u, padding_value=0).outputs
        # u_values: [#] [*=c]
        # u_valid: [#] [*=c]
        u_values_broadcast = C.swapaxes(C.sequence.broadcast_as(u_values, k))
        # u_values_broadcast: [#, n] [1, *=c]
        u_valid_broadcast = C.sequence.broadcast_as(C.reshape(u_valid, (1,), 1), k)
        # u_valid_broadcast: [#, n] [*=c, 1] ~ shape verified correct at his point

        # print("u_values_broadcast shape:", u_values_broadcast.shape)
        # print("abk shape:", a.shape, b.shape, k.shape)
        phi = window_weight(a, b, k, u_values_broadcast)
        # phi: [#, n] [*=c, 1]
        zero = C.constant(0)
        phi = C.element_select(u_valid_broadcast, phi, zero, name="phi")
        # phi: [#, n] [*=c, 1]
        attended = C.reduce_sum(phi * C.sequence.broadcast_as(encoded_unpacked, phi), axis=0)
        # [#, n] [1, char_ohe]
        # print("attended_context shape:", attended_context.shape)
        output = C.squeeze(attended, name="GaussianWindowAttention")
        # [#, n] [char_ohe]
        return output

    return attention


def PreTrainedBertEncoder(tf_bert_model_filepath: str, num_heads: int, dropout_rate: float = None):
    """ Use pre-trained tensorflow bert model

    Currently it is tested to work with:
        - `BERT-Base, Uncased`, uncased_L-12_H-768_A-12

    Models can be downloaded at https://github.com/google-research/bert

    Arguments:
        tf_bert_model_filepath (str): file path to the tensorflow model
        num_heads (int): number of attention heads in self attention
        dropout_rate (float): probability of dropping out an element in encoder

    Returns:
        :class:`~cntk.ops.functions.Function`:
        TF to CNTK Pre-trained Bert Encoder (Transformer Encoder)
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("Loading a TensorFlow models in CNTK, requires TensorFlow to be installed. Please see "
                          "https://www.tensorflow.org/install/ for installation instructions.")

    def bert_encoder_layer_number(layer_name: str, prefix):
        """ extracts 'xx' in '{prefix}{layer_xx/}{rest of the layer name}'

        There must not be any '/' on the left of layer_xx after prefix has been ignored.

        Arguments:
            layer_name (str): name of the layer
            prefix (str): prefix to be ignored

        Returns:
            int

        """
        # TODO: use regex probably more straightforward
        end_index = layer_name.index('/', len(prefix))
        number = int(layer_name[len(prefix):end_index].replace('layer_', ''))
        return number

    bert_encoder_prefix = 'bert/encoder/'

    variables_meta = tf.train.list_variables(tf_bert_model_filepath)
    encoder_variable_meta = [meta for meta in variables_meta if bert_encoder_prefix in meta[0]]

    layer_numbers = [bert_encoder_layer_number(meta[0], bert_encoder_prefix) for meta in encoder_variable_meta]
    nb_layers = max(layer_numbers) + 1  # +1 because layer numbering assumed to start from zero
    assert min(layer_numbers) == 0, f"Layer numbering assumed to start from zero but loaded model start from {min(layer_numbers)}"

    intermediate_dim = [meta[1][0] for meta in encoder_variable_meta if 'intermediate/dense/bias' in meta[0]]
    assert all(dim == intermediate_dim[0] for dim in intermediate_dim)
    intermediate_dim = intermediate_dim[0]

    model_dim = [meta[1][0] for meta in encoder_variable_meta if 'attention/output/dense/bias' in meta[0]]
    assert all(dim == model_dim[0] for dim in model_dim)
    model_dim = model_dim[0]

    encoder_layers = []
    mha_output_layernorm_bias_tag = 'attention/output/LayerNorm/beta'
    mha_output_layernorm_scale_tag = 'attention/output/LayerNorm/gamma'

    mha_output_dense_bias_tag = 'attention/output/dense/bias'
    mha_output_dense_kernel_tag = 'attention/output/dense/kernel'
    
    mha_key_bias_tag = 'attention/self/key/bias'
    mha_key_kernel_tag = 'attention/self/key/kernel'
    mha_query_bias_tag = 'attention/self/query/bias'
    mha_query_kernel_tag = 'attention/self/query/kernel'
    mha_value_bias_tag = 'attention/self/value/bias'
    mha_value_kernel_tag = 'attention/self/value/kernel'

    mha_dense_bias_tag = 'intermediate/dense/bias'
    mha_dense_kernel_tag = 'intermediate/dense/kernel'

    output_dense_bias_tag = 'output/dense/bias'
    output_dense_kernel_tag = 'output/dense/kernel'
    output_layernorm_scale_tag = 'output/LayerNorm/gamma'
    output_layernorm_bias_tag = 'output/LayerNorm/beta'

    config = {'num_heads': num_heads,
              'model_dim': model_dim,
              'intermediate_dim': intermediate_dim,
              'dropout_rate': dropout_rate,
              'obey_sequence_order': False,
              'max_seq_len': None,
              'key_init': mha_key_kernel_tag,
              'key_init_bias': mha_key_bias_tag,
              'query_init': mha_query_kernel_tag,
              'query_init_bias': mha_query_bias_tag,
              'value_init': mha_value_kernel_tag,
              'value_init_bias': mha_value_bias_tag,
              'mha_init': mha_output_dense_kernel_tag,
              'mha_init_bias': mha_output_dense_bias_tag,
              'mha_initial_scale': mha_output_layernorm_scale_tag,
              'mha_initial_bias': mha_output_layernorm_bias_tag,
              'intermediate_init': mha_dense_kernel_tag,
              'intermediate_init_bias': mha_dense_bias_tag,
              'init': output_dense_kernel_tag,
              'init_bias': output_dense_bias_tag,
              'initial_scale': output_layernorm_scale_tag,
              'initial_bias': output_layernorm_bias_tag,
              'name': None}
    
    for layer_num in range(nb_layers):
        prefix = f'bert/encoder/layer_{layer_num}/'
        initialised_config = {k: tf.train.load_variable(tf_bert_model_filepath, prefix + v) if isinstance(v, str) else v
                              for k, v in config.items()}

        initialised_config['name'] = f'encoder_layer_{layer_num}'
        encoder_layers.append(TransformerEncoderBlock(**initialised_config))

    @C.Function
    def model(x):

        for encoder_layer in encoder_layers:
            x = encoder_layer(x)

        return x

    return _inject_name(model, 'bert')


def PreTrainedBertModel(tf_bert_model_filepath: str, num_heads: int, dropout_rate: float = None):
    """ Initialise a pre-trained CNTK bert model converted from tensorflow

    Currently it is tested to work with:
        - `BERT-Base, Uncased`, uncased_L-12_H-768_A-12

    Models can be downloaded at https://github.com/google-research/bert

    Arguments:
        tf_bert_model_filepath (str): file path to the tensorflow model
        num_heads (int): number of attention heads in self attention
        dropout_rate (float): probability of dropping out an element in embedding and encoder

    Returns:
        :class:`~cntk.ops.functions.Function`:
        TF to CNTK Pre-trained Bert Model
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("Loading a TensorFlow models in CNTK, requires TensorFlow to be installed. Please see "
                          "https://www.tensorflow.org/install/ for installation instructions.")

    bert_embeddings = PreTrainedBertEmbeddings(tf_bert_model_filepath, dropout_rate)
    bert_encoder = PreTrainedBertEncoder(tf_bert_model_filepath, num_heads, 0.1)
    bert_pooler = PretrainedBertPooler(tf_bert_model_filepath)

    @C.Function
    def model(text_tensor, token_type_tensor):
        embedded = bert_embeddings(text_tensor, token_type_tensor)
        encoded = bert_encoder(embedded)
        pooled = bert_pooler(encoded)  # pooled is no longer a cntk sequence
        return pooled

    return model
