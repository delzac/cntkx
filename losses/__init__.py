import cntk as C
from cntkx.ops import gaussian_mdn_coeff
from math import pi


@C.typemap
def gaussian_mdn_loss(output_vector, target_vector, nmix: int, ndim: int):
    """
    Loss function for gaussian mixture density network. Usually used for regression problems.
    Mixture density networks are useful when trying to represent arbitrary conditional probabilities
    the same way a conventional neural network can represent arbitrary functions.

    Example:
        ndim, nmix = 1, 3
        input_tensor = C.input_variable(1, name="input_tensor")
        target_tensor = C.input_variable(1, name="target_tensor")

        # model
        inner = Dense(50, activation=C.relu)(input_tensor)
        inner = Dense(50, activation=C.relu)(inner)
        prediction_tensor = Dense((ndim + 2) * nmix, activation=None)(inner)

        loss = gaussian_mdn_loss(prediction_tensor, target_tensor, nmix=nmix, ndim=ndim)

    Arguments:
        output_vector: network output
        target_vector: ground truths (typically a continuous variable)
        nmix (int): number of mixtures
        ndim (int): number of dimensions in a gaussian kernel

    Returns:
        :class:`~cntk.ops.functions.Function`
    """

    @C.typemap
    def gaussian_mdn_phi(target, mu, sigma, ndim: int):
        """
        Calculates phi between the target tensor and the network prediction
        Does not assumes independence between components of target.

        Arguments:
            target: target tensor with shape (ndim, )
            mu: means of gaussian mdn with shape (nmix, ndim)
            sigma: sigma of gaussian mdn
            nmix (int): number of mixtures
            ndim (int): number of dimensions in gaussian

        Returns:
            :class:`~cntk.ops.functions.Function`
        """
        if not len(mu.shape) == 2:
            raise ValueError("mu {0} must have shape (nmix, ndim)".format(mu.shape))

        t = C.expand_dims(target, axis=0)

        exp_term = C.exp(C.negate(C.square(C.reduce_l2(t - mu, axis=-1)) / (2 * C.square(sigma))))
        factor = C.reciprocal((2 * pi) ** (ndim / 2) * C.pow(sigma, ndim))
        return factor * exp_term

    alpha, mu, sigma = gaussian_mdn_coeff(output_vector, nmix=nmix, ndim=ndim)
    phi = gaussian_mdn_phi(target_vector, mu, sigma, ndim=ndim)
    loss = C.negate(C.log(C.clip(C.reduce_sum(alpha * phi, axis=0), 1e-10, 1e10)))
    return loss
