import cntk as C
import numpy as np
from cntk.layers import Dense, LayerNormalization, ResNetBlock, Recurrence


def ScaledDotProductAttention(obey_sequence_order: bool = None, max_seq_len: int = None):
    """
    Scaled dot-product attention implementation of "Attention is all you need", https://arxiv.org/abs/1706.03762

    An attention function can be described as mapping a query and a set of key-value pairs to an output,
    where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
    of the values, where the weight assigned to each value is computed by a compatibility function of the
    query with the corresponding key.

    scaled_dot_product_attention(Q, K, V) = softmax(QV.T / sqrt(dk)) * V

    When query, key and value are all the same, it becomes self-attention.

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

    @C.Function
    def attention(query, key, value):
        dk = C.reduce_sum(C.ones_like(query))  # cannot use sequence.last, will conflict with recurrence

        unpacked_key = C.sequence.unpack(key, 0, True)  # [#] [-3, key_dim]
        unpacked_value = C.sequence.unpack(value, 0, True)  # [#] [-3, value_dim]

        broadcasted_key = C.sequence.broadcast_as(unpacked_key, query)  # [#, *] [-3, key_dim]
        scaled = C.times_transpose(query, broadcasted_key) / dk  # [#, *] [-3, key_dim]

        if obey_sequence_order and max_seq_len:
            # [#] [-3, -3], [#] [-3,]
            unpacked_scaled, scaled_mask = C.sequence.unpack(scaled, padding_value=0).outputs

            minus_inf = C.constant(-1e+30)
            valid_connections = C.Constant(np.tril(np.ones((max_seq_len, max_seq_len)), k=0))  # [] [max_seq, max_seq]
            valid_connections = C.reconcile_dynamic_axes(valid_connections, unpacked_scaled)  # [#] [max_seq, max_seq]
            valid_connections = C.crop_manual(valid_connections, unpacked_scaled, 0, 0)  # [#] [-3, -3]
            unpacked_scaled = C.element_select(valid_connections, unpacked_scaled, minus_inf)  # [#] [-3, -3]
            scaled = C.to_sequence_like(unpacked_scaled, query)  # [#, *] [-3]

        elif obey_sequence_order and not max_seq_len:
            raise ValueError("max_seq_len must be defined when obey_sequence_order is True")

        attended = C.times(C.softmax(scaled), C.sequence.broadcast_as(unpacked_value, query))  # [#, *] [value_dim,]
        return attended

    return attention


def MultiHeadAttention(num_heads, model_dim, obey_sequence_order: bool = None, max_seq_len: int = None):
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

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    assert model_dim % num_heads == 0, "Model dimension must be divisible by number of heads"

    head_dim = int(model_dim / num_heads)
    query_linears = [Dense(head_dim) for __ in range(num_heads)]
    key_linears = [Dense(head_dim) for __ in range(num_heads)]
    value_linears = [Dense(head_dim) for __ in range(num_heads)]
    multihead_liner = Dense(model_dim)
    sdpa = ScaledDotProductAttention(obey_sequence_order, max_seq_len)

    @C.Function
    def inner(query, key, value):
        # list of num_heads heads with shape (-3, head_dim) each
        attention_outputs = [sdpa(q_linear(query), k_linear(key), v_linear(value))
                             for q_linear, k_linear, v_linear in zip(query_linears, key_linears, value_linears)]

        result = multihead_liner(C.splice(*attention_outputs))
        return result

    return inner


def MultiHeadAttentionBlock(num_heads, model_dim, obey_sequence_order: bool = None, max_seq_len: int = None):
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

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    attention_layer = MultiHeadAttention(num_heads, model_dim, obey_sequence_order, max_seq_len)
    layernorm = LayerNormalization()

    @C.Function
    def block(query, key, value):
        attended = attention_layer(query, key, value)
        skip_connect_attended = attended + query
        normed_skip_connect_attended = layernorm(skip_connect_attended)
        return normed_skip_connect_attended

    return block


def TransformerEncoderBlock(num_heads: int, model_dim: int, obey_sequence_order: bool = None, max_seq_len: int = None):
    """ Encoder block of transformer as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Consist of 1 multi head attention followed by a dense layer, residual connect and layer norm

    Arguments:
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    mha_block = MultiHeadAttentionBlock(num_heads, model_dim, obey_sequence_order, max_seq_len)
    feed_foward = Dense(model_dim)
    layernorm = LayerNormalization()

    @C.Function
    def block(x):
        inner = mha_block(x, x, x)
        output = layernorm(ResNetBlock(feed_foward)(inner))
        return output

    return block


def TransformerDecoderBlock(num_heads: int, model_dim: int, obey_sequence_order: bool = None, max_seq_len: int = None):
    """ Decoder block of transformer as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Consist of 2 multi head attention followed by a dense layer, residual connect and layer norm

    Arguments:
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    mha_block1 = MultiHeadAttentionBlock(num_heads, model_dim, obey_sequence_order, max_seq_len)
    mha_block2 = MultiHeadAttentionBlock(num_heads, model_dim, obey_sequence_order, max_seq_len)

    feed_foward = Dense(model_dim)
    layernorm_feed_foward = LayerNormalization()

    @C.Function
    def block(encoded, x):
        inner = mha_block1(x, x, x)
        inner = mha_block2(inner, encoded, encoded)
        output = layernorm_feed_foward(ResNetBlock(feed_foward)(inner))
        return output

    return block


def TransformerEncoder(n: int, num_heads: int, model_dim: int, obey_sequence_order: bool = None, max_seq_len: int = None):
    """ Transformer encoder as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Example:
        a = C.sequence.input_variable(10)
        encoded = TransformerDecoder(3, 2, 10)(a)

        assert encoded.shape == (10, )

    Arguments:
        n (int): number of encoder blocks
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """

    blocks = [TransformerEncoderBlock(num_heads, model_dim, obey_sequence_order, max_seq_len)
              for __ in range(n)]

    @C.Function
    def inner(x):

        for block in blocks:
            x = block(x)

        return x

    return inner


def TransformerDecoder(n: int, num_heads: int, model_dim: int, obey_sequence_order: bool = None,
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
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """

    blocks = [TransformerDecoderBlock(num_heads, model_dim, obey_sequence_order, max_seq_len)
              for __ in range(n)]

    @C.Function
    def decoder(encoded, x):

        for block in blocks:
            x = block(encoded, x)

        return x

    return decoder


def Transformer(num_encoder_blocks: int = 6, num_decoder_blocks=6, num_heads_encoder: int = 16,
                num_heads_decoder: int = 16, encoder_model_dim: int = 512, decoder_model_dim: int = 512,
                encoder_obey_sequence_order: bool = False, decoder_obey_sequence_order: bool = True,
                max_seq_len_encoder: int = None, max_seq_len_decoder: int = 100):
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
        encoder_obey_sequence_order: if to allow self-attention to peek into future elements in sequence Default False.
        decoder_obey_sequence_order: if to allow self-attention to peak into future elements in sequence. Default True.
        max_seq_len_encoder: max sequence length in encoding sequence. Used for preventing attention peeking into future values.
        max_seq_len_decoder: max sequence length in decoding sequence. Used for preventing attention peeking into future values.

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    encoder = TransformerEncoder(n=num_encoder_blocks, num_heads=num_heads_encoder, model_dim=encoder_model_dim,
                                 obey_sequence_order=encoder_obey_sequence_order, max_seq_len=max_seq_len_encoder)

    decoder = TransformerDecoder(n=num_decoder_blocks, num_heads=num_heads_decoder, model_dim=decoder_model_dim,
                                 obey_sequence_order=decoder_obey_sequence_order, max_seq_len=max_seq_len_decoder)

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

    def sequence_position(seq):

        @C.Function
        def count(a, b):
            return a + 1 + b * 0

        return Recurrence(count)(C.slice(seq, 0, 0, 1))  # {#, *] [1,]

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
        u = sequence_position(encoded)
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
