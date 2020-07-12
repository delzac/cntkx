import cntk as C
import cntkx as Cx
import numpy as np
from cntk.layers import ResNetBlock
from cntkx.layers import PreTrainedBertEmbeddings, PositionwiseFeedForward, Dense, PretrainedBertPooler, Recurrence
from cntkx.layers import LayerNormalization
from cntk.default_options import default_override_or
from cntk.layers.blocks import _inject_name


def LinearAttention(hidden_dim: int, model_dim: int,
                    key_init=default_override_or(C.glorot_uniform()), key_init_bias=default_override_or(0),
                    query_init=default_override_or(C.glorot_uniform()), query_init_bias=default_override_or(0),
                    value_init=default_override_or(C.glorot_uniform()), value_init_bias=default_override_or(0),
                    name=''):
    """ Attention model that is linear in time and memory complexity.
    This is a huge improvement from standard softmax attention models or self-attention
    where the time and memory complexity is quadratic in sequence length.

    This is especially significant since cntk doesn't have any build-in checkpointing functionality
    that saves gpu memory and hence allow the training of Transformer models. With this attention,
    it becomes possible to do transformer training on cntk.

    This implementation addresses the limitation of attentions by express the attention
    as a linear dot-product of kernel feature maps and made use of the associativity property of matrix products.

    When query, key and value are all the same, it becomes self-attention.

    For more details refer to "Transformers are RNNs:Fast Autoregressive Transformers with Linear Attention" by
    Katharopoulos et al. (https://arxiv.org/abs/2006.16236)

    Note:
        Key and value must have the same sequence length

    Example:
        a = C.sequence.input_variable(24)
        b = LinearAttention(hidden_dim=32, model_dim=24)(a, a, a)

        assert b.shape == (32, )

    Arguments:
        hidden_dim (int): number of dim in final output, does of projection of Value
        model_dim (int): number of dim in the attention
        key_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        key_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        query_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        query_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        value_init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
        value_init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    query_linear = Dense(model_dim, init=query_init, init_bias=query_init_bias)
    key_linear = Dense(model_dim, init=key_init, init_bias=key_init_bias)
    value_linear = Dense(hidden_dim, init=value_init, init_bias=value_init_bias)

    def phi(x):  # kernel
        return C.elu(x) + 1

    @C.BlockFunction('LinearAttention', name=name)
    def model(query, key, value):
        q = phi(query_linear(query))
        k = phi(key_linear(key))
        v = value_linear(value)

        # key and value should have the same sequence length
        k_unpacked = C.sequence.unpack(k, padding_value=0, no_mask_output=True)
        # k_unpacked: [#] [*kv=, model_dim]
        v_unpacked = C.sequence.unpack(v, padding_value=0, no_mask_output=True)
        # v_unpacked: [#] [*kv=, hidden_dim]
        kv = C.times(C.swapaxes(k_unpacked), v_unpacked)
        # kv [#] [model_dim, hidden_dim]
        kv_broadcasted = C.sequence.broadcast_as(kv, q)  # this can be reused across queries
        # kv [#, *] [model_dim, hidden_dim]

        numerator = C.squeeze(C.times(C.expand_dims(q, axis=C.Axis.new_leading_axis()), kv_broadcasted))
        # numerator [#, *] [hidden_dim, ]
        denom = C.reduce_sum(q * C.sequence.broadcast_as(C.sequence.reduce_sum(k), q))
        # denom [#, *] [1]

        return numerator / denom

    return model


def LinearAttentionModel(hidden_dim: int, model_dim: int,
                         key_init=default_override_or(C.glorot_uniform()), key_init_bias=default_override_or(0),
                         query_init=default_override_or(C.glorot_uniform()), query_init_bias=default_override_or(0),
                         value_init=default_override_or(C.glorot_uniform()), value_init_bias=default_override_or(0),
                         name=''):
    """ Convenience wrapper in the style of cntk.layers.AttentionModel """
    attention = LinearAttention(hidden_dim=hidden_dim, model_dim=model_dim,
                                key_init=key_init, key_init_bias=key_init_bias,
                                query_init=query_init, query_init_bias=query_init_bias,
                                value_init=value_init, value_init_bias=value_init_bias, name=name)

    def model(encoder_hidden_state, decoder_hidden_state):
        return attention(decoder_hidden_state, encoder_hidden_state, encoder_hidden_state)

    return model


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

    Arguments:
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed

    Returns:
        :class:`~cntk.ops.functions.Function`:
        A function that returns a weighted sum of value

    """

    def attention(query, key, value):
        return Cx.scaled_dot_product_attention(query, key, value, obey_sequence_order, max_seq_len, name)

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


def GaussianWindowAttention(nb_mixtures, activation=C.softplus, init=C.he_normal(), name=''):
    """
    Implementation of the attention model found in "Generating sequences with recurrent neural networks" by Alex Graves.

    Gaussian window attention uses a directional mixture of gaussian kernels as convolution/attention window.

    For more details, the paper can be found in https://arxiv.org/abs/1308.0850

    Note:
        There is a slight deviation from the original implementation where we use softplus as the activation
        function instead of exp. Exp activation causes some minor instability.
    
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
    dense = Dense(shape=3 * nb_mixtures, activation=activation, init=init, name="GravesAttention")

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
        a = C.slice(abk, 0, 0, nb_mixtures)
        b = C.slice(abk, 0, nb_mixtures, 2 * nb_mixtures)
        k = C.slice(abk, 0, 2 * nb_mixtures, 0)
        k = Recurrence(C.plus)(k)

        a = C.expand_dims(a, axis=-1)
        b = C.expand_dims(b, axis=-1)
        k = C.expand_dims(k, axis=-1)
        return a, b, k

    @C.BlockFunction('GaussianWindowAttention', name)
    def attention(encoded, network):
        abk = dense(network)
        a, b, k = gaussian_windows_attention_coefficients(abk, nb_mixtures)
        # print("abk shape:", a.shape, b.shape, k.shape)
        # a, b, k: [#, n] [nb_mixture, 1]
        # context: [#, c] [char_ohe]

        encoded_unpacked = C.sequence.unpack(encoded, padding_value=0, no_mask_output=True)
        # context_unpacked: [#] [*=c, char_ohe]
        u = Cx.sequence.position(encoded)  # position gives shape=(1, )
        # u: [#, c], [1]
        u_values, u_valid = C.sequence.unpack(u, padding_value=999_999).outputs
        # u_values: [#] [*=c, 1]
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


def GaussianAttentionSeqImage(n: int, image_height: int, expected_image_width: int, name=''):
    """ Gaussian attention applied to an encoded sequence image (i.e. sequence axis is image width)

    This implementation is from the deepmind paper, DRAW: A Recurrent Neural Network for Image Generation by Gregor et al
    More details can be found in the following https://arxiv.org/abs/1502.04623

    Example:
        n = 5
        num_channels = 3
        image_height = 64
        expected_image_width = 1000
        image_seq = C.sequence.input_variable((num_channels, image_height))  # rgb image with variable width and fixed height
        decoder_hidden_state = ...  # from decoder somewhere in the network
        attended_image = Cx.layers.GaussianAttentionSeqImage(n, image_height, expected_image_width)(image_seq, decoder_hidden_state)

        assert attended_image.shape == (num_channels, n, n)

    Arguments:
        n (int): number of gaussian attention filter per grid dimension,
          where total of number of attention filter = n * n grid
        image_height (int): the static image height dimension of the sequence
        expected_image_width (int): Expected number of cols (width) in the image

    """
    dense = Dense(shape=(5, ))
    A = expected_image_width
    B = image_height

    def attention_parameters(network_outputs):
        g_x = 0.5 * (A + 1) * (network_outputs[0] + 1)  # grid centre - x (cols)
        g_y = 0.5 * (B + 1) * (network_outputs[1] + 1)  # grid centre - y (rows)
        sigma2 = C.exp(network_outputs[2])  # isotropic variance
        delta = (max(A, B) - 1) / (n - 1) * C.exp(network_outputs[3])  # stride
        gamma = C.exp(network_outputs[4])  # intensity
        return g_x, g_y, sigma2, delta, gamma

    @C.BlockFunction('GaussianAttentionSeqImage', name)
    def model(seq_image, decoded):
        params = dense(decoded)
        g_x, g_y, sigma2, delta, gamma = attention_parameters(params)

        i = C.Constant(np.arange(n) + 1,)  # col of patch
        j = C.Constant(np.arange(n) + 1,)  # row of patch
        mu_x = g_x + (i - n / 2 - 0.5) * delta
        mu_y = g_y + (j - n / 2 - 0.5) * delta
        mu_x = C.expand_dims(mu_x, axis=-1)
        mu_y = C.expand_dims(mu_y, axis=-1)
        # mu_x: [#, *] [n, 1]
        # mu_y: [#, *] [n, 1]

        image = C.sequence.unpack(seq_image, padding_value=0, no_mask_output=True)
        # image: [#] [*image_width, filters, image_height]

        width_pos = Cx.sequence.position(seq_image)
        # width_pos: [#, *] [1]

        width_pos_unpacked = C.sequence.unpack(width_pos, padding_value=999_999, no_mask_output=True)
        # width_pos: [#] [*image_width, 1]

        a = C.sequence.broadcast_as(C.swapaxes(width_pos_unpacked), mu_x)
        # a: [#, *] [1, *image_width]
        # x pos index of image (width)

        b = C.Constant(np.arange(image_height).reshape((1, -1)))
        # b: [] [1, image_height]
        # y pos index of image (height)

        # calculate the which portion of the image that is attended by the gaussian filter
        f_xi = C.exp(-0.5 * C.square(a - mu_x) / sigma2)
        f_yj = C.exp(-0.5 * C.square(b - mu_y) / sigma2)
        # f_xi: [#, *] [n, *image_width]
        # f_yj: [#, *] [n, image_height]

        z_x = C.reduce_sum(f_xi, axis=1)
        z_y = C.reduce_sum(f_yj, axis=1)
        # z_x: [#, *] [n]
        # z_y: [#, *] [n]

        f_xi = f_xi / z_x
        f_yj = f_yj / z_y
        # f_xi: [#, *] [n, *image_width]
        # f_yj: [#, *] [n, image_height]

        # combine filters from x and y
        image_broadcasted = C.sequence.broadcast_as(image, f_yj)
        attended = gamma * C.times(f_xi, C.times_transpose(image_broadcasted, f_yj), output_rank=2)
        # attended: [#, *] [n, filters, n]
        attended = C.swapaxes(attended)
        # attended: [#, *] [filters, n (x) , n (y)]
        return attended

    return model


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
