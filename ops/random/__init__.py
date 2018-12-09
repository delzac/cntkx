import cntk as C


@C.Function
def sample(x):
    """ Sample a distribution and returns a one hot encode vector

    Arguments:
        x: input tensor

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    perturbed_x = x + C.random.gumbel_like(x)
    return C.hardmax(perturbed_x)
