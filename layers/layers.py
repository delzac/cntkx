import cntk as C
import cntkx as Cx
from cntk.layers import SequentialConvolution, Recurrence, Dense, LayerNormalization, ResNetBlock
from cntkx.ops import scaled_dot_product_attention


def QRNN(window: int=1, hidden_dim=None, activation=C.tanh, return_full_state=False):
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

    Arguments:
        window (`int`):  Defines the size of the convolutional window (how many previous
          tokens to look when computing the QRNN values). Defaults 1.
        hidden_dim (int): size of hidden dim of h, c and o
        activation: cell activation function

    Returns:
        :class:`~cntk.ops.functions.Function`: OR
        tuple of :class:`~cntk.ops.functions.Function`:

    """

    def FoPool(c, fz):
        f = C.slice(fz, 0, 0, hidden_dim)
        z = C.slice(fz, 0, hidden_dim, 2 * hidden_dim)
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

        # FoPooling
        c = Recurrence(FoPool)(C.splice(f, z))
        h = o * c

        if return_full_state:
            return h, c
        else:
            return h

    return model


def MultiheadAttention(num_heads, model_dim, map_ranks: tuple = None, obey_sequence_order: bool = None,
                       max_seq_len: int = None, output_as_seq: bool = False):
    """ Multi-head attention as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Example:
        a = C.sequence.input_variable(10)
        b = MultiheadAttention(2, 10)(a, a, a, None)

        assert b.shape == (10, )

    Arguments:
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        map_ranks (tuple): map_rank for query, key, value if tuple, else qkv will get same map_ranks.
          set 1 if input tensor is an unpacked sequence, None if it is a sequence. Default None.
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed
        output_as_seq: output attended tensor as a sequence

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    assert model_dim % num_heads == 0, "Model dimension must be divisible by number of heads"
    map_ranks = (map_ranks, map_ranks, map_ranks) if not isinstance(map_ranks, tuple) else map_ranks
    map_rank_query, map_rank_key, map_rank_value = map_ranks

    head_dim = int(model_dim / num_heads)
    query_linears = [Dense(head_dim, map_rank=map_rank_query) for __ in range(num_heads)]
    key_linears = [Dense(head_dim, map_rank=map_rank_key) for __ in range(num_heads)]
    value_linears = [Dense(head_dim, map_rank=map_rank_value) for __ in range(num_heads)]
    multihead_liner = Dense(model_dim, map_rank=0 if output_as_seq else 1)

    def inner(query, key, value, dynamic_axes_like=None):
        # list of num_heads heads with shape (-3, head_dim) each
        attention_outputs = [scaled_dot_product_attention(q_linear(query), k_linear(key), v_linear(value),
                                                          dynamic_axes_like, obey_sequence_order, max_seq_len,
                                                          output_as_seq)
                             for q_linear, k_linear, v_linear in zip(query_linears, key_linears, value_linears)]

        result = multihead_liner(C.splice(*attention_outputs))
        return result

    return inner


def MultiHeadAttentionBlock(num_heads, model_dim, map_ranks: tuple = None, obey_sequence_order: bool = None,
                            max_seq_len: int = None, output_as_seq: bool = False):
    """ Multi head attention block as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Multi-head attention block comes with a residual connection and a layer norm.

    Example:
        a = C.sequence.input_variable(10)
        b = MultiHeadAttentionBlock(2, 10)(a, a, a, None)

        assert b.shape == (10, )

    Arguments:
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        map_ranks (tuple): first item is for query/key. Second is value. set 1 if input tensor
          is an unpacked sequence, None if it is a sequence. Default None.
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed
        output_as_seq: output attended tensor as a sequence

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    attention_layer = MultiheadAttention(num_heads, model_dim, map_ranks, obey_sequence_order, max_seq_len, output_as_seq)
    layernorm = LayerNormalization()

    def block(query, key, value, dynamic_axes_like=None):
        # TODO: skip connect input is not always value, setting as query will be correct
        dynamic_seq_axis_present = any(ax.is_sequence_axis for ax in query.dynamic_axes)

        if dynamic_seq_axis_present and output_as_seq:
            skip_connect_input = query
        elif dynamic_seq_axis_present and not output_as_seq:
            skip_connect_input = C.sequence.unpack(query, padding_value=0, no_mask_output=True)
        elif not dynamic_seq_axis_present and output_as_seq:
            skip_connect_input = C.to_sequence_like(query, dynamic_axes_like)
        elif not dynamic_seq_axis_present and not output_as_seq:
            skip_connect_input = query
        else:
            raise ValueError("This branch should not be reachable")

        attended = attention_layer(query, key, value, dynamic_axes_like)
        skip_connect_attended = attended + skip_connect_input
        normed_skip_connect_attended = layernorm(skip_connect_attended)
        return normed_skip_connect_attended

    return block


def TransformerEncoderBlock(num_heads: int, model_dim: int, map_rank=None,
                            obey_sequence_order: bool = None, max_seq_len: int = None, output_as_seq: bool = False):
    """ Encoder block of transformer as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Consist of 1 multi head attention followed by a dense layer, residual connect and layer norm

    Arguments:
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        map_rank: 1 if input_tensor is an unpacked sequence, None if sequence. Default None.
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed
        output_as_seq: output attended tensor as a sequence

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    mha_block = MultiHeadAttentionBlock(num_heads, model_dim, map_rank, obey_sequence_order, max_seq_len, output_as_seq)
    feed_foward = Dense(model_dim, map_rank=0 if output_as_seq else 1)
    layernorm = LayerNormalization()

    def block(x, dynamic_axes_like=None):
        inner = mha_block(x, x, x, dynamic_axes_like)
        output = layernorm(ResNetBlock(feed_foward)(inner))
        return output

    return block


def TransformerDecoderBlock(num_heads: int, model_dim: int, is_encoded_seq: bool, map_rank=None,
                            obey_sequence_order: bool = None, max_seq_len: int = None, output_as_seq: bool = False):
    """ Decoder block of transformer as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Consist of 2 multi head attention followed by a dense layer, residual connect and layer norm

    Arguments:
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        is_encoded_seq (bool): is encoded tensor a sequence
        map_rank: '1' if input tensor x is an unpacked sequence, 'None' if sequence. Default None.
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed
        output_as_seq: output attended tensor as a sequence

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    encoded_map_rank = (1, None if is_encoded_seq else 1, None if is_encoded_seq else 1)
    mha_block1 = MultiHeadAttentionBlock(num_heads, model_dim, map_rank, obey_sequence_order, max_seq_len, output_as_seq=False)
    mha_block2 = MultiHeadAttentionBlock(num_heads, model_dim, encoded_map_rank, obey_sequence_order, max_seq_len, output_as_seq)

    feed_foward = Dense(model_dim, map_rank=0 if output_as_seq else 1)
    layernorm_feed_foward = LayerNormalization()

    def block(encoded, x, dynamic_axes_like=None):
        dynamic_seq_axis_present = any(ax.is_sequence_axis for ax in x.dynamic_axes)

        dynamic_axes_like2 = dynamic_axes_like
        if dynamic_seq_axis_present and dynamic_axes_like is None:
            dynamic_axes_like2 = x

        inner = mha_block1(x, x, x, dynamic_axes_like)
        inner = mha_block2(inner, encoded, encoded, dynamic_axes_like2)
        output = layernorm_feed_foward(ResNetBlock(feed_foward)(inner))
        return output

    return block
