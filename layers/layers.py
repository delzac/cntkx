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


def MultiheadAttention(nb_heads, model_dim, map_rank=None, obey_sequence_order: bool = None,
                       max_seq_len: int = None, output_as_seq: bool = False):
    """ Multi head attention as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Arguments:
        nb_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        map_rank (int, None): If input is an unpacked sequence tensor set as 1. Default None.
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed
        output_as_seq: output attended tensor as a sequence

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    assert model_dim % nb_heads == 0, "Model dimension must be divisible by number of heads"

    head_dim = int(model_dim / nb_heads)
    query_linears = [Dense(head_dim, map_rank=map_rank) for __ in range(nb_heads)]
    key_linears = [Dense(head_dim, map_rank=map_rank) for __ in range(nb_heads)]
    value_linears = [Dense(head_dim, map_rank=map_rank) for __ in range(nb_heads)]
    multihead_liner = Dense(model_dim, map_rank=0 if output_as_seq else 1)

    def inner(query, key, value, dynamic_axes_like=None):
        # list of nb_heads heads with shape (-3, head_dim) each
        attention_outputs = [scaled_dot_product_attention(q_linear(query), k_linear(key), v_linear(value),
                                                          dynamic_axes_like, obey_sequence_order, max_seq_len,
                                                          output_as_seq)
                             for q_linear, k_linear, v_linear in zip(query_linears, key_linears, value_linears)]

        result = multihead_liner(C.splice(*attention_outputs))
        return result

    return inner


def TransformerEncoderBlock(nb_heads: int, model_dim: int, map_rank=None,
                            obey_sequence_order: bool = None, max_seq_len: int = None, output_as_seq: bool = False):
    """ Encoder block of transformer as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Arguments:
        nb_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        map_rank (int, None): If input is an unpacked sequence tensor set as 1. Default None.
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed
        output_as_seq: output attended tensor as a sequence

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    attention_layer = MultiheadAttention(nb_heads, model_dim, map_rank, obey_sequence_order, max_seq_len, output_as_seq)
    layernorm_attention = LayerNormalization()
    feed_foward = Dense(model_dim, map_rank=0 if output_as_seq else 1)
    layernorm_feed_foward = LayerNormalization()

    def encoder(q, k, v, dynamic_axes_like=None):
        dynamic_seq_axis_present = any(ax.is_sequence_axis for ax in v.dynamic_axes)

        if dynamic_seq_axis_present and output_as_seq:
            skip_connecet_input = v
        elif dynamic_seq_axis_present and not output_as_seq:
            skip_connecet_input = C.sequence.unpack(v, padding_value=0, no_mask_output=True)
        elif not dynamic_seq_axis_present and output_as_seq:
            skip_connecet_input = C.to_sequence_like(v, dynamic_axes_like)
        elif not dynamic_seq_axis_present and not output_as_seq:
            skip_connecet_input = v
        else:
            raise ValueError("This branch should not be reachable")

        attended = attention_layer(q, k, v, dynamic_axes_like)
        skip_connect_attended = attended + skip_connecet_input
        normed_skip_connect_attended = layernorm_attention(skip_connect_attended)
        output = layernorm_feed_foward(ResNetBlock(feed_foward)(normed_skip_connect_attended))
        return output

    return encoder


def TransformerEncoder(n: int, nb_heads: int, model_dim: int, obey_sequence_order: bool = None,
                       max_seq_len: int = None, output_as_seq: bool = False):
    """ Transformer encoder as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Arguments:
        n (int): number of encoder blocks
        nb_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed
        output_as_seq: output attended tensor as a sequence

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """

    if n >= 2:
        first = [TransformerEncoderBlock(nb_heads, model_dim, None, obey_sequence_order, max_seq_len, False)]
        last = [TransformerEncoderBlock(nb_heads, model_dim, 1, obey_sequence_order, max_seq_len, output_as_seq)]
        mid = [TransformerEncoderBlock(nb_heads, model_dim, 1, obey_sequence_order, max_seq_len, False)
               for __ in range(n - 2)]

        blocks = first + mid + last if mid else first + last

    elif n == 1:
        blocks = [TransformerEncoderBlock(nb_heads, model_dim, None, obey_sequence_order, max_seq_len, output_as_seq)]
    else:
        raise ValueError(f"n ({n}) must be larger than 0")

    def inner(x):

        seq = x
        x = blocks.pop(0)(x, x, x, None)

        for block in blocks:
            x = block(x, x, x, seq)

        return x

    return inner


def TransformerDecoderBlock():

    def decoder():
        return None

    return decoder
