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


@C.typemap
def focal_loss_with_softmax(output_vector, target_vector, alpha=1, gamma=2., axis=-1, name=''):
    """
    CNTK multi-class implementation of focal loss from "Focal Loss for Dense Object Detection" by Tsung-Yi Lin et al.

    Focal loss add a factor (1 - p) ^ gamma to the standard cross entropy criterion. Setting gamma > 0 reduces the
    relative loss for well-classified examples (p > .5), putting more focus on hard, misclassified examples.
    Focal loss enables the training of highly accurate dense object detectors in the presence of vast
    numbers of easy background examples or dataset with extreme class imbalance (e.g. 1:1000).

    This implementation will work in semantic segmentation of images i.e. output can
    be a rank 2 tensor of shape (num_classes, row, col)

    Maths:
        Focal Loss = - alpha * (1 - p) ^ gamma * log ( p )

    Example:
        Cx.focal_loss_with_softmax([[0, 0, 0.8, 0.2]], [[0, 0, 1, 0]]).eval()
        array([[0.31306446]], dtype=float32)

    Arguments:
        output_vector: the unscaled computed output values from the network. Can be
          from shape (num_classes,) for classification up to shape (num_classes, row, col) for semantic segmentation
          of images.
        target_vector: usually it is one-hot vector where the hot bit
         corresponds to the label index. But it can be any probability
         distribution over the labels.
        alpha (float): sacling factor. weight assigned to rare classes.
          should slightly decrease as gamma increase. (defaults 1)
        gamma (float): Larger gamma reduces relative loss for well-classified examples.
          Recommended range [0.5, 5] (Default 2.)
        axis (int or :class:`~cntk.axis.Axis`, optional): if given, focal loss will be computed
                along this axis
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    prob = C.softmax(output_vector, axis=axis)
    log_prob = target_vector * C.log(prob)  # cross entropy with softmax

    factor = C.pow(1 - prob, gamma)

    return C.negate(alpha * C.reduce_sum(factor * log_prob, axis=axis), name=name)


@C.typemap
def binary_focal_loss(output, target, alpha=1., gamma=2., name=''):
    """
    CNTK binary class implementation of focal loss from "Focal Loss for Dense Object Detection" by Tsung-Yi Lin et al.

    Focal loss add a factor (1 - p) ^ gamma to the standard cross entropy criterion. Setting gamma > 0 reduces the
    relative loss for well-classified examples (p > .5), putting more focus on hard, misclassified examples.
    Focal loss enables the training of highly ccurate dense object detectors in the presence of vast
    numbers of easy background examples or dataset with extreme class imbalance (e.g. 1:1000).

    This implementation will work in semantic segmentation of images i.e. output can
    be a rank 2 tensor of shape (row, col). Output will be correct even in edge case where entire image is background.

    Maths:
        Focal Loss = - alpha * (1 - p) ^ gamma * log ( p )

    Arguments:
        output: the computed posterior probability from the network (typ. a ``sigmoid``). Can be
          from shape (1,) for simple classification up to shape (row, col) for semantic segmentation of images.
        target: ground-truth label, 0 or 1
        alpha (float): sacling factor. weight assigned to rare classes.
          should slightly decrease as gamma increase. (defaults 1)
        gamma (float): Larger gamma reduces relative loss for well-classified examples.
          Recommended range [0.5, 5] (Default 2.)
        axis (int or :class:`~cntk.axis.Axis`, optional): if given, focal loss will be computed
                along this axis
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    logprobA = target * C.log(output)
    logprobB = (1 - target) * C.log(1 - output)

    factorA = C.pow(1 - output, gamma)
    factorB = C.pow(output, gamma)

    return C.negate(alpha * (factorA * logprobA + factorB * logprobB), name=name)


def cross_entropy_with_softmax(output_vector, target_vector, axis=-1, label_smoothing=0., name=''):
    """ Adds label smoothing regularisation to C.cross_entropy_with_softmax

    Label smoothing regularisation prevents the largest logit from becoming much larger than all others.

    Classical cross entropy maximises log-likelihood of the correct label. This can cause overfitting.
    If the model learns to assign full probability to the ground-truth label for each training example, it
    is not guaranteed to generalize. Second, it encourages the differences between the largest logit and all others to
    become large, and this, combined with the bounded gradient, reduces the ability of the model to adapt.
    Intuitively, this happens because the model becomes too confident about its predictions.

    To solve this, we introduce label smoothing.

    For more details on label smoothing, please refer to Rethinking the Inception Architecture for Computer Vision
    by Szegedy et al. (https://arxiv.org/abs/1512.00567)

    Arguments:
        output_vector: the unscaled computed output values from the network
        target_vector: usually it is one-hot vector where the hot bit
         corresponds to the label index. But it can be any probability
         distribution over the labels.
        axis (int or :class:`~cntk.axis.Axis`, optional): if given, cross entropy will be computed
                along this axis
        label_smoothing (float): a small label smoothing constant. As a guideline, in ImageNet training of 1000 classes,
            label_smoothing can be set to 0.1
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    if label_smoothing == 0.:
        return C.cross_entropy_with_softmax(output_vector, target_vector, axis=axis, name=name)
    else:
        k = output_vector.shape[axis]
        target_vector = (1 - label_smoothing) * target_vector + label_smoothing / k
        return C.cross_entropy_with_softmax(output_vector, target_vector, axis=axis, name=name)


def generalised_robust_barron_loss(output_vector, target_vector, alpha: float, scale: float):
    """ Generalised robust loss that can be used as replacement to either l1 or L2 norm or any regression task.

    By introducing robustness as a continuous parameter, our loss function allows algorithms
    built around robust loss minimization to be generalized, which improves performance on
    basic vision tasks such as registration and clustering.

    This implements the rho(x, \alpha, c) function described in
    "A General and Adaptive Robust Loss Function", Jonathan T. Barron, https://arxiv.org/abs/1701.03077
    View is video on this talk https://www.youtube.com/watch?v=BmNKbnF69eY

    The approximate form of the loss which is faster to compute, but inaccurate
    when the difference between `output_vector` and `target_vector`  and alpha are near zero.

    Different alpha allows you to get various forms of robust loss:
        alpha = 2: L2 loss
        alpha = 1: pseudo-Huber loss
        alpha = 0: Cauchy loss
        alpha = -2: Geman-McClure loss
        alpha = -inf: Welsch loss

    The effect of outlier diminishes as alpha becomes more negative, and as alpha approaches -inf an
    outlier whose residual magnitude is larger than 3 * scale is almost completely ignored.

    This generalised loss for all alpha has the following properties:
        Monotonic (useful for annealing)
        Smooth wrt inputs and alpha
        Bounded first and second derivatives

    Note:
        this is NOT the adaptive implementation of barron loss, which is slightly more complicated to implement,
        since it uses cubic spline interpolation to estimate the partition function.

        I may implement it once i have time.

    Arguments:
        output_vector: tensor (assumes last axis is the dimension axis where the loss is to be computed from)
        target_vector: tensor
        alpha (float): controls the form that the loss takes.The effect of outlier diminishes as alpha becomes more
          negative, and as alpha approaches -inf an outlier whose residual magnitude is larger than 3 * scale
          is almost completely ignored.
        scale (float): controls the size of the loss's quadratic bowl near x = 0, the derivative of the loss is
          approximately linear when |x| < scale.

    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    # alpha = C.Parameter(shape=(output_vector.shape[-1],), init=C.glorot_uniform())
    # scale = C.Parameter(shape=(output_vector.shape[-1],), init=C.glorot_uniform())
    # alpha_negative = C.less(alpha, 0)
    #
    # x = output_vector - target_vector
    #
    # b = C.abs(alpha - 2) + epsilon
    # d = C.element_select(alpha_negative, alpha - epsilon, alpha + epsilon)
    # loss = (b / d) * (C.pow(C.square(x / scale) / b + 1., 0.5 * d) - 1.)

    x = output_vector - target_vector

    # Avoids the 3 singularity at alpha == 2, alpha == 0, and alpha == -inf
    if alpha == 2:
        loss = C.square(x / scale)  # L2 loss
    elif alpha == 0:
        loss = C.log(0.5 * C.square(x / scale) + 1)  # Cauchy loss
    elif alpha <= -1e8:
        loss = 1 - C.exp(-0.5 * C.square(x / scale))  # welsch loss
    else:
        alpha_offset = abs(alpha - 2)
        factor = alpha_offset / alpha

        core = C.pow(C.square(x / scale) / alpha_offset + 1, alpha / 2)
        loss = factor * (core - 1)

    return loss
