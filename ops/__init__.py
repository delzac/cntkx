import cntk as C
import numpy as np
from . import sequence
from . import random
from cntk.layers.blocks import _inject_name


##########################################################################
# linear ops
##########################################################################
def scalar(x, name=''):
    """ select first element of x with shape (1,)

    Arguments:
        x: input tensor

    Returns:
        :class:`~cntk.ops.functions.Function`
        a scalar of shape (1,)
    """
    @C.BlockFunction('scalar', name)
    def inner(x):
        return C.slice(C.reshape(x, (-1,)), 0, 0, 1)

    return inner(x)


def cumsum(x, axis: int = -1):
    """ Calculates the cumulative sum across a static axis

    Arguments:
        x: input tensor
        axis (int): static axis of tensor to cumsum over

    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    d = x.shape[axis]
    u = C.constant(np.triu(np.ones((d, d))).astype(x.dtype))
    if axis != -1:
        x = C.swapaxes(x, -1, axis)
    z = C.times(x, u)
    if axis != -1:
        z = C.swapaxes(z, -1, axis)
    return z


def batchmatmul(left, right, output_rank=1, infer_input_rank_to_map=C.TIMES_NO_INFERRED_INPUT_RANK, name=''):
    """ Batch Matrix Multiplication

    The output of this operation is the matrix product of the two input batch matrices.

    This implementation is similar to tensorflow.matmul.

    Currently assumes the first axis to be the static batch axis. Does not accept multiple static batch axis.

    Example:
        a = C.sequence.input_variable((3, 4, 5))     # batch matrix
        b = C.sequence.input_variable((3, 5, 6))     # batch matrix
        c = Cx.batchmatmul(a, b)
        assert c.shape == (3, 4, 6)                  # 3 is treated as a batch axis


        a = C.sequence.input_variable((3, 4, 5))     # batch matrix
        b = C.sequence.input_variable((3, 5, 6, 7))  # batch tensor
        c = Cx.batchmatmul(a, b, output_rank=2)
        assert c.shape == (3, 4, 6, 7)               # 3 is treated as a batch axis


        a = C.input_variable((3, 4, 5))              # batch matrix
        b = C.input_variable((3, 5, 6, 7))           # batch tensor
        c = Cx.batchmatmul(a, b, output_rank=2)
        assert c.shape == (3, 4, 6, 7)


    Arguments:
        left: left side matrix or tensor
        right: right side matrix or tensor
        output_rank (int): in case we have tensors as arguments, output_rank represents
            the number of axes to be collapsed in order to transform the tensors
            into matrices, perform the operation and then reshape back (explode the axes)
        infer_input_rank_to_map (int): meant for internal use only. Always use default value
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    """

    left_shape = left.shape
    right_shape = right.shape

    seq_axis_present = len(left.dynamic_axes) == 2
    static_batch_axis = left_shape[0]  # assumes the first axis to be the static batch axis.

    if left_shape[0] != right_shape[0]:
        raise ValueError("first axis of left operand and right operand must be the same")

    if (left_shape[0] < 0 or right_shape[0] < 0) and seq_axis_present:
        raise ValueError("Static batch axis cannot be a free axis when dynamic sequence axis is also present")

    # Combine dynamic sequence axis and static batch axis
    if not seq_axis_present:
        left_unpacked = left
        right_unpacked = right
    else:
        left_unpacked = C.sequence.unpack(left, padding_value=0, no_mask_output=True)
        right_unpacked = C.sequence.unpack(right, padding_value=0, no_mask_output=True)

        left_unpacked = C.reshape(left_unpacked, (-1,) + left_shape[1:])
        right_unpacked = C.reshape(right_unpacked, (-1,) + right_shape[1:])

    # Fold static batch axis into dynamic sequence axis
    left_folded = C.to_sequence(left_unpacked)  # do not set sequence length as batch axis has been folded in
    right_folded = C.to_sequence_like(right_unpacked, left_folded)  # seq_length / axis set here to tell cntk they have the same seq axis

    # Matrix Multiply when no static batch axis is present
    result = C.times(left_folded, right_folded, output_rank=output_rank, infer_input_rank_to_map=infer_input_rank_to_map)

    # Split dynamic sequence axis back to original dynamic sequence and static batch axis
    result_unpacked = C.sequence.unpack(result, padding_value=0, no_mask_output=True)
    if not seq_axis_present:
        result_packed = C.reshape(result_unpacked, (static_batch_axis, ) + result.shape)
    else:
        result_unfolded = C.reshape(result_unpacked, (-1, static_batch_axis) + result.shape)
        result_packed = C.to_sequence_like(result_unfolded, left)

    return _inject_name(result_packed, name)


def upsample(x):
    """ Up sample image by a factor of 2 using nearest neighbour.

    Example:
        a = C.input_variable((3, 32, 32)
        b = UpSampling2D(a)

        assert b.shape == (3, 64, 64)

    Arguments:
        x: input image tensor, assumed (channel, row, col)

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    xr = C.reshape(x, (x.shape[0], x.shape[1], 1, x.shape[2], 1))
    xx = C.splice(xr, xr, axis=-1)  # axis=-1 refers to the last axis
    xy = C.splice(xx, xx, axis=-3)  # axis=-3 refers to the middle axis
    r = C.reshape(xy, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2))
    return r


def centre_crop(larger_image, smaller_image, name: str = ''):
    """ Centre crop spatial dimensions only.

    Arguments:
        larger_image: class:`~cntk.ops.functions.Function` that outputs the tensor to be centre cropped
        smaller_image: class:`~cntk.ops.functions.Function` that outputs the reference tensor
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    input_shape = larger_image.shape  # larger
    referent_shape = smaller_image.shape  # smaller
    row_offset = int((input_shape[1] - referent_shape[1]) / 2)
    col_offset = int((input_shape[2] - referent_shape[2]) / 2)

    if row_offset == 0 and col_offset == 0:
        return larger_image

    elif row_offset < 0 or col_offset < 0:
        raise ValueError(f"offset became negative, check if image was passed correctly. "
                         f"larger image {larger_image.shape}, smaller image {smaller_image.shape}")

    return C.crop_manual(larger_image, smaller_image, row_offset, col_offset, name=name)


def centre_crop_and_splice(larger_image, smaller_image):
    """ Implementation of copy and crop found in UNET architecture.

    Arguments:
        larger_image: to be centre cropped and channel spliced into smaller image
        smaller_image: reference tensor

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    return C.splice(smaller_image, centre_crop(larger_image, smaller_image), axis=0)


##########################################################################
# non linear and nn ops
##########################################################################
@C.typemap
def swish(x, name=''):
    """ swish activation function first introduced in 'Searching for activation function' by Prajit et al.
    Paper can be found in https://arxiv.org/abs/1710.05941 and https://arxiv.org/abs/1901.02671

    It typically exhibits good performance in a variety of task in vision and nlp problems.
    Can be used as a drop-in replace for relu.
    """

    @C.BlockFunction('Swish', name=name)
    def inner(a):
        return a * C.sigmoid(a)

    return inner(x)


@C.typemap
def hardmax(x, axis=-1, name=''):
    """
    This hardmax implementation can be applied on selected axis. Original cntk hardmax can only be applied on all axis.

    If ``axis`` is given as integer, then the hardmax will be computed along that axis.
    If the provided ``axis`` is -1, it will be computed along the last axis. if None, it will be applied to all axes.

    Arguments:
        x: input_tensor
        axis (int or :class:`~cntk.axis.Axis`): axis along which the hardmax operation will be performed
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`:
    """

    @C.BlockFunction('Hardmax', name=name)
    def inner(a):
        return C.equal(C.reduce_max(a, axis=axis), a)

    return inner(x)


def erf(x, name=''):
    """
    Computes the element-wise error function of `x`:

    The output tensor has the same shape as ``x``.

    This implementation is from the Handbook of Mathematical Functions and
    has error less than 1.5 * 10-7 for all inputs.
    book can be found here 'http://people.math.sfu.ca/~cbm/aands/frameindex.htm'

    """

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    @C.BlockFunction('Erf', name=name)
    def inner(a):
        not_negative = C.greater_equal(a, 0)
        sign = C.element_select(not_negative, not_negative, -1)

        abs_x = C.abs(a)

        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * a)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * C.exp(-abs_x * abs_x)
        return C.element_times(sign, y)

    return inner(x)


def gelu(x, name=''):
    """ Gaussian Error Linear Unit (GELU), a high-performing neuralnetwork activation function.
    The GELU nonlinearity is the expected transforma-tion of a stochastic regularizer which randomly
    applies the identity or zero mapto a neuron’s input.  The GELU nonlinearity weights inputs by their
    magnitude,rather than gates inputs by their sign as in ReLUs.

    For more detail please refer to 'Gaussian Error Linear Units (GELU)'
    by Hendrycks and Gimpel (https://arxiv.org/abs/1606.08415)

    This activation is used in BERT and OpenAI GPT & GPT-2.

    Its computationally x2 times slower than relu with some negligible increase in memory footprint.

    Arguments:
        x: input_tensor
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    @C.BlockFunction('Gelu', name=name)
    def inner(a):
        return 0.5 * a * (1 + erf(a / 1.41421356237))

    return inner(x)


def gelu_fast(x, name=''):
    """ This version is an less good approximation of gelu but it is x2 times faster on GPU and x3.8 faster on CPU.
    This implementation just as fast as relu on GPU but x2 slower on CPU.

    Roughly the same memory footprint as relu.

    Arguments:
        x: input_tensor
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    @C.BlockFunction('GeluFast', name=name)
    def inner(a):
        return a * C.sigmoid(1.702 * a)

    return inner(x)


##########################################################################
# mixture density network ops
##########################################################################
@C.typemap
def gaussian_mdn_coeff(x, nmix: int, ndim: int):
    """
    Extracts the coefficients for gaussian mixture density network.
    Assumes independence between gaussian dimensions.

    Example:
        ndim, nmix = 1, 3
        a = C.input_variable(ndim)
        prediction = Dense((ndim + 2) * nmix)(a)
        coeffs = C.combine(gaussian_mdn_coeff(prediction_tensor, nmix=nmix, ndim=ndim)).eval({a: x})

        alpha, mu, sigma = coeffs.values()

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

    Example:
        ndim, nmix = 1, 3
        a = C.input_variable(ndim)
        prediction = Dense((ndim + 2) * nmix)(a)
        sampled = sample_gaussian_mdn(prediction, nmix, ndim)

        results = sampled.eval({a: x})  # different results every time you eval

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
