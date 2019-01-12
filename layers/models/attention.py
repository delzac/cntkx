from cntkx.layers import TransformerEncoderBlock, TransformerDecoderBlock


def TransformerEncoder(n: int, num_heads: int, model_dim: int, obey_sequence_order: bool = None,
                       max_seq_len: int = None, output_as_seq: bool = False):
    """ Transformer encoder as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Example:
        a = C.sequence.input_variable(10)

        encoded = TransformerDecoder(3, 2, 10)(a)

        assert encoded.shape == (-3, 10)

    Arguments:
        n (int): number of encoder blocks
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed
        output_as_seq: output attended tensor as a sequence

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """

    if n >= 2:
        first = [TransformerEncoderBlock(num_heads, model_dim, None, obey_sequence_order, max_seq_len, False)]
        last = [TransformerEncoderBlock(num_heads, model_dim, 1, obey_sequence_order, max_seq_len, output_as_seq)]
        mid = [TransformerEncoderBlock(num_heads, model_dim, 1, obey_sequence_order, max_seq_len, False)
               for __ in range(n - 2)]

        blocks = first + mid + last if mid else first + last

    elif n == 1:
        blocks = [TransformerEncoderBlock(num_heads, model_dim, None, obey_sequence_order, max_seq_len, output_as_seq)]
    else:
        raise ValueError(f"n ({n}) must be larger than 0")

    def inner(x):

        seq = x
        x = blocks.pop(0)(x, None)

        for block in blocks:
            x = block(x, seq)

        return x

    return inner


def TransformerDecoder(n: int, num_heads: int, model_dim: int, is_encoded_seq: bool, obey_sequence_order: bool = None,
                       max_seq_len: int = None, output_as_seq: bool = False):
    """ Transformer decoder as described in "Attention is all you need", https://arxiv.org/abs/1706.03762

    Example:
        a = C.sequence.input_variable(10)
        encoded = C.input_variable((-1, 10)

        decoded = TransformerDecoder(3, 2, 10, is_encoded_seq=False)(encoded, a)

        assert decoded.shape == (-3, 10)

    Arguments:
        n (int): number of decoder blocks
        num_heads (int): number of attention heads
        model_dim (int): number of hidden dim in final output of multi-head attention
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed
        output_as_seq: output attended tensor as a sequence

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """

    if n >= 2:
        first = [TransformerDecoderBlock(num_heads, model_dim, is_encoded_seq, None, obey_sequence_order, max_seq_len, False)]
        last = [TransformerDecoderBlock(num_heads, model_dim, is_encoded_seq, 1, obey_sequence_order, max_seq_len, output_as_seq)]
        mid = [TransformerDecoderBlock(num_heads, model_dim, is_encoded_seq, 1, obey_sequence_order, max_seq_len, False)
               for __ in range(n - 2)]

        blocks = first + mid + last if mid else first + last

    elif n == 1:
        blocks = [TransformerDecoderBlock(num_heads, model_dim, is_encoded_seq, None, obey_sequence_order, max_seq_len, output_as_seq)]
    else:
        raise ValueError(f"n ({n}) must be larger than 0")

    def decoder(encoded, x):
        seq = x
        x = blocks.pop(0)(encoded, x, None)

        for block in blocks:
            x = block(encoded, x, seq)

        return x

    return decoder


def Transformer(num_encoder_blocks: int = 6, num_decoder_blocks=6, num_heads_encoder: int = 16,
                num_heads_decoder: int = 16, model_dim: int = 512, encoder_obey_sequence_order: bool = False,
                decoder_obey_sequence_order: bool = True, max_seq_len_encoder: int = None,
                max_seq_len_decoder: int = 100, output_as_seq: bool = True):
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
        model_dim: model output dimension (should be the same dimension as the transformer input)
        encoder_obey_sequence_order: if to allow self-attention to peek into future elements in sequence Default False.
        decoder_obey_sequence_order: if to allow self-attention to peak into future elements in sequence. Default True.
        max_seq_len_encoder: max sequence length in encoding sequence. Used for preventing attention peeking into future values.
        max_seq_len_decoder: max sequence length in decoding sequence. Used for preventing attention peeking into future values.
        output_as_seq: transformer outputs as a sequence or unpacked tensor (no dynamic sequence axis)

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    encoder = TransformerEncoder(n=num_encoder_blocks, num_heads=num_heads_encoder, model_dim=model_dim,
                                 obey_sequence_order=encoder_obey_sequence_order, max_seq_len=max_seq_len_encoder,
                                 output_as_seq=False)

    decoder = TransformerDecoder(n=num_decoder_blocks, num_heads=num_heads_decoder, model_dim=model_dim,
                                 is_encoded_seq=False, obey_sequence_order=decoder_obey_sequence_order,
                                 max_seq_len=max_seq_len_decoder, output_as_seq=output_as_seq)

    def model(tensor_to_encode, decoder_input_tensor):
        encoded = encoder(tensor_to_encode)
        decoded = decoder(encoded, decoder_input_tensor)
        return decoded

    return model
