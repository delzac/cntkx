import cntk as C
import numpy as np
from . import sequence
from . import random


##########################################################################
# linear ops
##########################################################################

def cumsum(x, axis: int=-1):
    """ Calculates the cumulative sum across a static axis

    Arguments:
        x: input tensor
        axis (int): static axis of tensor to cumsum over

    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    d = x.shape[axis]
    u = C.constant(np.triu(np.ones((d,d))).astype(x.dtype))
    if axis != -1:
        x = C.swapaxes(x, -1, axis)
    z = C.times(x, u)
    if axis != -1:
        z = C.swapaxes(z, -1, axis)
    return z


##########################################################################
# non linear ops
##########################################################################
def scaled_dot_product_attention(query, key, value, valid_mask_value=None, obey_sequence_order: bool = None,
                                 max_seq_len: int = None, output_as_seq: bool = False):
    """
    Scaled dot-product attention implementation of "Attention is all you need", https://arxiv.org/abs/1706.03762

    An attention function can be described as mapping a query and a set of key-value pairs to an output,
    where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
    of the values, where the weight assigned to each value is computed by a compatibility function of the
    query with the corresponding key.

    attention(Q, K, V) = softmax(QV.T / sqrt(dk)) * V

    When query, key and value are all the same, it becomes self-attention.

    Arguments:
        query: input tensor of rank 2 or a sequence of rank 1 tensor (i.e. vector)
        key: input tensor of rank 2 or a sequence of rank 1 tensor (i.e. vector)
        value: input tensor of rank 2 or a sequence of rank 1 tensor (i.e. vector)
        valid_mask_value: a tensor with values 1 or 0 of shape (seq_len, ). Used to select out sequences that had been padded.
        obey_sequence_order: do not let attention peek into future values
        max_seq_len: max sequence length possible, used to ensure that sequence order is obeyed
        output_as_seq: output attended tensor as a sequence

    Returns:
        :class:`~cntk.ops.functions.Function`: weighted sum of value

    """
    dyanmic_seq_axis_present = any(ax.is_sequence_axis for ax in value.dynamic_axes)
    dk = sum(i for i in key.shape if i > 0)

    unpacked_key = key
    unpacked_query = query
    unpacked_value = value

    if dyanmic_seq_axis_present and valid_mask_value is None:
        unpacked_key, valid_mask_key = C.sequence.unpack(key, padding_value=0).outputs
        unpacked_query, valid_mask_query = C.sequence.unpack(query, padding_value=0).outputs
        unpacked_value, valid_mask_value = C.sequence.unpack(value, padding_value=0).outputs

    elif dyanmic_seq_axis_present and valid_mask_value:
        raise ValueError("valid_mask_value must be None if value has dynamic sequence axis present")

    scaled = C.times_transpose(unpacked_query, unpacked_key) / dk  # [#] [*, *] seq_len x seq_len

    if obey_sequence_order and max_seq_len:
        minus_inf = C.constant(-1e+30)
        valid_connections = C.Constant(np.tril(np.ones((max_seq_len, max_seq_len)), k=0))
        valid_connections = C.reconcile_dynamic_axes(valid_connections, scaled)
        valid_connections = C.crop_manual(valid_connections, scaled, 0, 0)
        scaled = C.element_select(valid_connections, scaled, minus_inf)

    elif obey_sequence_order and not max_seq_len:
        raise ValueError("max_seq_len must be defined when obey_sequence_order is True")

    attended = C.times(C.softmax(scaled), unpacked_value)

    if output_as_seq and dyanmic_seq_axis_present:
        attended = C.to_sequence_like(attended, value)
    elif output_as_seq and not dyanmic_seq_axis_present and valid_mask_value:
        attended = C.to_sequence(attended, C.reduce_sum(valid_mask_value))
    elif not output_as_seq:
        attended = C.element_select(C.expand_dims(valid_mask_value, -1), attended, C.Constant(0))
    else:
        raise ValueError(f"This should not happen output_as_seq={output_as_seq}")

    return attended


##########################################################################
# mixture density network ops
##########################################################################


@C.typemap
def gaussian_mdn_coeff(x, nmix: int, ndim: int):
    """
    Extracts the coefficients for gaussian mixture density network.
    Assumes independence between gaussian dimensions.

    Arguments:
        x: input tensor
        nmix (int): number of mixture
        ndim (int): number of dimension of gaussian

    Returns:
        tuple

    """

    if len(x.shape) != 1:
        raise ValueError("Must be a 1d tensor, but input has shape {0}".format(x.shape))

    alpha = C.softmax(C.slice(x, 0, 0, nmix), name='alpha')
    sigma = C.exp(C.slice(x, 0, nmix, 2 * nmix), name='sigma')  # common variance for all components in single gaussian kernel
    mu = C.reshape(C.slice(x, 0,  2 * nmix, (ndim + 2) * nmix), shape=(nmix, ndim), name='mu')
    return alpha, mu, sigma


def sample_gaussian_mdn(prediction_tensor, nmix: int, ndim: int):
    """ Constructs sampling nodes from mixture density network outputs

    Arguments:
        prediction_tensor: input tensor
        nmix (int): number of mixture
        ndim (int): number of dimension of gaussian

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    alpha_tensor, mu_tensor, sigma_tensor = gaussian_mdn_coeff(prediction_tensor, nmix=nmix, ndim=ndim)

    selected_alpha = random.sample(alpha_tensor)
    selected_mu_tensor = C.reduce_sum(mu_tensor * C.expand_dims(selected_alpha, axis=-1), axis=0)
    selected_sigma_tensor = C.reduce_sum(sigma_tensor * selected_alpha, axis=0)

    sampled = C.random.normal_like(selected_sigma_tensor) * selected_sigma_tensor + selected_mu_tensor
    return sampled
