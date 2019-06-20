import cntk as C


@C.typemap
def sample(x, axis=-1, name=''):
    """ Sample an unnormalised log-prob distribution and returns a one hot encode vector

    Arguments:
        x: input tensor
        name (str): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    perturbed_x = x + C.random.gumbel_like(x)
    return C.equal(C.reduce_max(perturbed_x, axis=axis), perturbed_x, name=name)  # equivalent to hardmax(perturbed_x)


@C.typemap
def sample_top_k(x, k, num_classes, axis=-1, name=''):
    """ Sample once from the top_k unnormalised log-prob distribution of `x` and returns a one hot encoded vector.

    Example:
        import cntk as C
        import cntkx as Cx

        a = C.input_variable(5)
        b = Cx.random.sample_top_k(a, k=3, num_classes=5)

        n = np.array([[1, 2, 3, 4, 5],] * 1000)

        results = b.eval({a: n})
        assert np.sum(results[:, :2]) == 0
        assert np.sum(results[:, 2:]) == 1000


    Arguments:
        x: input tensor
        k (int): number of k largest probability to sample from
        num_classes (int): typically the dimension of `x`
        axis (int): axis along which to perform the operation (default: -1)
        name (str): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    # x: [#, *] [static_axes, num_classes]

    k_values, k_indices = C.top_k(x, k=k, axis=axis).outputs
    # k_indices [#, *] [static_axes, k]

    b = C.one_hot(k_indices, num_classes)
    # b: [#, *] [static_axes, k, num_classes]

    valid_probabilities = C.reduce_sum(b, axis=-2, keepdims=False)
    # valid_probabilities: [#, *] [static_axes, num_classes]

    # k largest probabilies are retained, everything else is set to -inf and will not be sampled
    minus_inf = C.constant(-1e+30)
    d = x * valid_probabilities
    e = C.element_select(d, d, minus_inf)
    # e: [#, *] [static_axes, num_classes]

    # sample from top_k distribution once
    s = sample(e, axis=axis, name=name)
    # s: [#, *] [static_axes, num_classes]

    return s
