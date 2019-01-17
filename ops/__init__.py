import cntk as C
import numpy as np
from . import sequence
from . import random


##########################################################################
# linear ops
##########################################################################

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
    return C.equal(C.reduce_max(x, axis=axis), x, name=name)


@C.typemap
def erf(x, name=''):
    """
    Computes the element-wise error function of `x`:

    The output tensor has the same shape as ``x``.

    This implementation is from the Handbook of Mathematical Functions and
    has error less than 1.5 * 10-7 for all inputs.
    book can be found here 'http://people.math.sfu.ca/~cbm/aands/frameindex.htm'

    """
    not_negative = C.greater_equal(x, 0)
    sign = C.element_select(not_negative, not_negative, -1)

    abs_x = C.abs(x)

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * C.exp(-abs_x * abs_x)
    return C.element_times(sign, y, name=name)


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
