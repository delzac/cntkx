# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

'''
Basic building blocks that are semantically not layers (not used in a layered fashion),
e.g. the LSTM block.
'''

from __future__ import division
import numpy as np
import cntk as C
from cntk.layers import Stabilizer, SentinelValueForAutoSelectRandomSeed
from cntk.variables import Constant, Parameter
from cntk.ops import times, slice, sigmoid, tanh, relu
from cntk.internal import _as_tuple
from cntk.initializer import glorot_uniform
from cntk.default_options import get_default_override, default_override_or

_INFERRED = (C.InferredDimension,)  # as a tuple, makes life easier

from cntk.ops.functions import BlockFunction  # (deprecated)


def _RecurrentBlock(type, shape, cell_shape, activation, use_peepholes,
                    init, init_bias,
                    enable_self_stabilization, dropout_rate, seed,
                    name=''):
    '''
    Helper to create a recurrent block of type 'WeightDroppedLSTM', 'GRU', or RNNStep.
    '''

    has_projection = cell_shape is not None

    shape = _as_tuple(shape)

    cell_shape = _as_tuple(cell_shape) if cell_shape is not None else shape
    if len(shape) != 1 or len(cell_shape) != 1:
        raise ValueError("%s: shape and cell_shape must be vectors (rank-1 tensors)" % type)
        # otherwise we'd need to fix slicing and Param initializers

    stack_axis = -1  # for efficient computation, we stack multiple variables (along the fastest-changing one, to match BS)
    # determine stacking dimensions
    cell_shape_list = list(cell_shape)
    stacked_dim = cell_shape_list[stack_axis]
    cell_shape_list[stack_axis] = stacked_dim * {
        'IndRNN': 1,
        'IndyLSTM': 4,
        'WeightDroppedLSTM': 4
    }[type]
    cell_shape_stacked = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times
    cell_shape_list[stack_axis] = stacked_dim * {
        'IndRNN': 1,
        'IndyLSTM': 4,
        'WeightDroppedLSTM': 4
    }[type]
    cell_shape_stacked_H = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times

    # parameters
    b  = Parameter(            cell_shape_stacked,   init=init_bias, name='b')                              # bias
    W  = Parameter(_INFERRED + cell_shape_stacked,   init=init,      name='W')                              # input
    H  = Parameter(shape     + cell_shape_stacked_H, init=init,      name='H')                              # hidden-to-hidden
    H1 = Parameter(            cell_shape_stacked_H, init=init,      name='H1') if type == 'IndyLSTM' else None  # hidden-to-hidden
    H2 = Parameter(shape                 ,           init=init,      name='H2') if type == 'IndRNN' else None  # hidden-to-hidden
    Ci = Parameter(            cell_shape,           init=init,      name='Ci') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
    Cf = Parameter(            cell_shape,           init=init,      name='Cf') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
    Co = Parameter(            cell_shape,           init=init,      name='Co') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}

    Wmr = Parameter(cell_shape + shape, init=init, name='P') if has_projection else None  # final projection

    # each use of a stabilizer layer must get its own instance
    Sdh = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='dh_stabilizer')
    Sdc = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='dc_stabilizer')
    Sct = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='c_stabilizer')
    Sht = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='P_stabilizer')

    # DropConnect
    dropout = C.layers.Dropout(dropout_rate=dropout_rate, seed=seed, name='h_dropout')

    # define the model function itself
    # general interface for Recurrence():
    #   (all previous outputs delayed, input) --> (outputs and state)
    # where
    #  - the first output is the main output, e.g. 'h' for LSTM
    #  - the remaining outputs, if any, are additional state
    #  - if for some reason output != state, then output is still fed back and should just be ignored by the recurrent block

    # LSTM model function
    # in this case:
    #   (dh, dc, x) --> (h, c)
    def weight_dropped_lstm(dh, dc, x):

        dhs = Sdh(dh)  # previous values, stabilized
        dcs = Sdc(dc)
        # note: input does not get a stabilizer here, user is meant to do that outside

        # projected contribution from input(s), hidden, and bias
        proj4 = b + times(x, W) + times(dhs, dropout(H))

        it_proj  = slice (proj4, stack_axis, 0*stacked_dim, 1*stacked_dim)  # split along stack_axis
        bit_proj = slice (proj4, stack_axis, 1*stacked_dim, 2*stacked_dim)
        ft_proj  = slice (proj4, stack_axis, 2*stacked_dim, 3*stacked_dim)
        ot_proj  = slice (proj4, stack_axis, 3*stacked_dim, 4*stacked_dim)

        # helper to inject peephole connection if requested
        def peep(x, c, C):
            return x + C * c if use_peepholes else x

        it = sigmoid(peep(it_proj, dcs, Ci))  # input gate(t)
        # TODO: should both activations be replaced?
        bit = it * activation(bit_proj)  # applied to tanh of input network

        ft = sigmoid(peep(ft_proj, dcs, Cf))  # forget-me-not gate(t)
        bft = ft * dc  # applied to cell(t-1)

        ct = bft + bit  # c(t) is sum of both

        ot = sigmoid(peep(ot_proj, Sct(ct), Co))  # output gate(t)
        ht = ot * activation(ct)  # applied to tanh(cell(t))

        c = ct  # cell value
        h = times(Sht(ht), Wmr) if has_projection else ht

        return h, c

    # LSTM model function
    # in this case:
    #   (dh, dc, x) --> (h, c)
    def indy_lstm(dh, dc, x):

        dhs = Sdh(dh)  # previous values, stabilized
        dcs = Sdc(dc)
        # note: input does not get a stabilizer here, user is meant to do that outside

        # projected contribution from input(s), hidden, and bias
        proj4 = b + times(x, W) + C.splice(dhs, dhs, dhs, dhs) * H1  # 4 is the number of stacked dim

        it_proj  = slice (proj4, stack_axis, 0*stacked_dim, 1*stacked_dim)  # split along stack_axis
        bit_proj = slice (proj4, stack_axis, 1*stacked_dim, 2*stacked_dim)
        ft_proj  = slice (proj4, stack_axis, 2*stacked_dim, 3*stacked_dim)
        ot_proj  = slice (proj4, stack_axis, 3*stacked_dim, 4*stacked_dim)

        # helper to inject peephole connection if requested
        def peep(x, c, C):
            return x + C * c if use_peepholes else x

        it = sigmoid(peep(it_proj, dcs, Ci))  # input gate(t)
        # TODO: should both activations be replaced?
        bit = it * activation(bit_proj)  # applied to tanh of input network

        ft = sigmoid(peep(ft_proj, dcs, Cf))  # forget-me-not gate(t)
        bft = ft * dc  # applied to cell(t-1)

        ct = bft + bit  # c(t) is sum of both

        ot = sigmoid(peep(ot_proj, Sct(ct), Co))  # output gate(t)
        ht = ot * activation(ct)  # applied to tanh(cell(t))

        c = ct  # cell value
        h = times(Sht(ht), Wmr) if has_projection else ht

        return h, c

    def ind_rnn(dh, x):
        dhs = Sdh(dh)  # previous value, stabilized
        ht = activation(times(x, W) + dhs * H2 + b)
        h = times(Sht(ht), Wmr) if has_projection else ht
        return h

    function = {
        'IndRNN': ind_rnn,
        'IndyLSTM': indy_lstm,
        'WeightDroppedLSTM':    weight_dropped_lstm
    }[type]

    # return the corresponding lambda as a CNTK Function
    return BlockFunction(type, name)(function)


def WeightDroppedLSTM(shape, dropout_rate, cell_shape=None, activation=default_override_or(tanh), use_peepholes=default_override_or(False),
         init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
         enable_self_stabilization=default_override_or(False), seed=SentinelValueForAutoSelectRandomSeed,
         name=''):
    '''
    WDLSTM(shape, cell_shape=None, activation=tanh, use_peepholes=False, init=glorot_uniform(), init_bias=0, enable_self_stabilization=False, name='')

    Layer factory function to create an LSTM block for use inside a recurrence.
    The LSTM block implements one step of the recurrence and is stateless. It accepts the previous state as its first two arguments,
    and outputs its new state as a two-valued tuple ``(h,c)``.

    Example:
     >>> # a typical recurrent LSTM layer
     >>> from cntkx.layers import *
     >>> lstm_layer = Recurrence(WeightDroppedLSTM(500))

    Args:
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        cell_shape (tuple, defaults to `None`): if given, then the output state is first computed at `cell_shape`
         and linearly projected to `shape`
        activation (:class:`~cntk.ops.functions.Function`, defaults to :func:`~cntk.ops.tanh`): function to apply at the end, e.g. `relu`
        use_peepholes (bool, defaults to `False`):
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to `glorot_uniform`): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        enable_self_stabilization (bool, defaults to `False`): if `True` then add a :func:`~cntk.layers.blocks.Stabilizer`
         to all state-related projections (but not the data input)
        name (str, defaults to ''): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`:
        A function ``(prev_h, prev_c, input) -> (h, c)`` that implements one step of a recurrent LSTM layer.
    '''

    activation                = get_default_override(WeightDroppedLSTM, activation=activation)
    use_peepholes             = get_default_override(WeightDroppedLSTM, use_peepholes=use_peepholes)
    init                      = get_default_override(WeightDroppedLSTM, init=init)
    init_bias                 = get_default_override(WeightDroppedLSTM, init_bias=init_bias)
    enable_self_stabilization = get_default_override(WeightDroppedLSTM, enable_self_stabilization=enable_self_stabilization)

    return _RecurrentBlock('WeightDroppedLSTM', shape, cell_shape, activation=activation, use_peepholes=use_peepholes,
                           init=init, init_bias=init_bias, dropout_rate=dropout_rate, seed=seed,
                           enable_self_stabilization=enable_self_stabilization, name=name)


# TODO: change activation default to relu and implement weight constraint
def IndRNN(shape, activation=default_override_or(relu),
            init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
            enable_self_stabilization=default_override_or(False), name=''):
    """
    IndRNN implementation found in "Independently Recurrent Neural Network (IndRNN): Building A Longer andDeeper RNN"
    by Li, et al (https://arxiv.org/abs/1803.04831).

    IndRNN are RNNS where neurons in each layer are independent from each other, and the cross-channel information is
    obtained through stacking multiple layers.

    It has been shown that an IndRNN can be easily regulated to prevent the gradient exploding and vanishing problems
    while allowing the networkto learn long-term dependencies. Moreover, an IndRNN can work with non-saturated
    activation functions such as relu (rectified linear unit) and be still trained robustly.
    Multiple IndRNNs can be stacked to construct a network that is deeper than the existing RNNs.
    Experimental results have shown that the proposed IndRNN is able to process very long
    sequences (over 5000 time steps), can be used to construct very deep networks (21 layers used in the experiment)
    and still be trained robustly. Better performances have been achieved on various tasks by using IndRNNs compared
    with the traditional RNN and LSTM.

    IndRNN also enables the usable of Relu activation which more efficient to compute than sigmoid and leads to
    faster convergence during training. You may consider to initialise the recurrent weights using a uniform
    distribution from 0 to 1.

    The original code is available at: https://github.com/Sunnydreamrain/IndRNN_Theano_Lasagne.

    Example:
     >>> # a plain relu RNN layer
     >>> from cntkx.layers import *
     >>> relu_rnn_layer = Recurrence(IndRNN(500))

    Args:
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        activation (:class:`~cntk.ops.functions.Function`, defaults to signmoid): function to apply at the end, e.g. `relu`
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to `glorot_uniform`): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        enable_self_stabilization (bool, defaults to `False`): if `True` then add a :func:`~cntk.layers.blocks.Stabilizer`
         to all state-related projections (but not the data input)
        name (str, defaults to ''): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`:
        A function ``(prev_h, input) -> h`` where ``h = activation(input @ W + prev_h * R + b)``
    """

    activation                = get_default_override(IndRNN, activation=activation)
    init                      = get_default_override(IndRNN, init=init)
    init_bias                 = get_default_override(IndRNN, init_bias=init_bias)
    enable_self_stabilization = get_default_override(IndRNN, enable_self_stabilization=enable_self_stabilization)

    return _RecurrentBlock('IndRNN', shape, None, activation=activation, use_peepholes=False,
                           init=init, init_bias=init_bias, dropout_rate=0, seed=SentinelValueForAutoSelectRandomSeed,
                           enable_self_stabilization=enable_self_stabilization, name=name)


def IndyLSTM(shape, activation=default_override_or(tanh),
            init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
            enable_self_stabilization=default_override_or(False), name=''):
    """
    Implementation of Independently Recurrent Long Short-term Memory cells: IndyLSTMs by Gonnet and Deselaers.
    Paper can be found at https://arxiv.org/abs/1903.08023

    IndyLSTM differ from regular LSTM cells in that the recurrent weights are not modeled as a full matrix,
    but as a diagonal matrix, i.e. the output and state of each LSTM cell depends on the inputs and its
    own output/state, as opposed to the input and the outputs/states of all the cells in the layer.
    The number of parameters per IndyLSTM layer, and thus the number of FLOPS per evaluation, is linear in the
    number of nodes in the layer, as opposed to quadratic for regular LSTM layers, resulting in potentially both
    smaller and faster model.

    Example:
     >>> # a gated recurrent layer
     >>> from cntkx.layers import *
     >>> indy_lstm_layer = Recurrence(IndyLSTM(500))

    Args:
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        cell_shape (tuple, defaults to `None`): if given, then the output state is first computed at `cell_shape`
         and linearly projected to `shape`
        activation (:class:`~cntk.ops.functions.Function`, defaults to :func:`~cntk.ops.tanh`): function to apply at the end, e.g. `relu`
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to `glorot_uniform`): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        enable_self_stabilization (bool, defaults to `False`): if `True` then add a :func:`~cntk.layers.blocks.Stabilizer`
         to all state-related projections (but not the data input)
        name (str, defaults to ''): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`:
        A function ``(prev_h, prev_c, input) -> (h, c)`` that implements one step of a recurrent IndyLSTM layer.
    """

    activation                = get_default_override(IndyLSTM, activation=activation)
    init                      = get_default_override(IndyLSTM, init=init)
    init_bias                 = get_default_override(IndyLSTM, init_bias=init_bias)
    enable_self_stabilization = get_default_override(IndyLSTM, enable_self_stabilization=enable_self_stabilization)

    return _RecurrentBlock('IndyLSTM', shape, None, activation=activation, use_peepholes=False,
                           init=init, init_bias=init_bias, dropout_rate=0, seed=SentinelValueForAutoSelectRandomSeed,
                           enable_self_stabilization=enable_self_stabilization, name=name)


def LSTM(shape, activation=default_override_or(tanh), weight_drop_rate=None,
         ih_init=default_override_or(glorot_uniform()), ih_bias=default_override_or(0),
         hh_init=default_override_or(glorot_uniform()), hh_bias=default_override_or(0),
         name=''):
    """ PyTorch style implementation of LSTM. Used for loading pytorch pretrained models.

    This difference between this implementation and cntk's one is that the slicing of
    the recurrent weights are different.

    pytorch is ifgo but cntk is igfo. And pytorch has 2 biases, but cntk only has one. In this implementation,
    i kept the biases to one to speed it up a little more.

    """
    activation = get_default_override(LSTM, activation=activation)
    ih_init = get_default_override(LSTM, ih_init=ih_init)
    ih_bias = get_default_override(LSTM, ih_bias=ih_bias)
    hh_init = get_default_override(LSTM, hh_init=hh_init)
    hh_bias = get_default_override(LSTM, hh_bias=hh_bias)

    stack_axis = - 1
    shape = _as_tuple(shape)
    cell_shape = shape
    cell_shape_list = list(cell_shape)
    stacked_dim = cell_shape_list[stack_axis]
    cell_shape_list[stack_axis] = stacked_dim * 4
    cell_shape_stacked = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times
    cell_shape_list[stack_axis] = stacked_dim * 4
    cell_shape_stacked_H = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times

    init_bias = ih_bias + hh_bias  # combine both biases in pytorch into one
    b  = Parameter(            cell_shape_stacked,   init=init_bias,    name='b')                    # bias
    W  = Parameter(_INFERRED + cell_shape_stacked,   init=ih_init,      name='W')                    # input
    H  = Parameter(shape     + cell_shape_stacked_H, init=hh_init,      name='H')                    # hidden-to-hidden

    dropout = C.layers.Dropout(dropout_rate=weight_drop_rate, name='h_dropout') if weight_drop_rate is not None else None

    @C.BlockFunction('PT::LSTM', name)
    def lstm(dh, dc, x):
        # projected contribution from input(s), hidden, and bias

        dropped_H = dropout(H) if weight_drop_rate is not None else H
        proj4 = b + times(x, W) + times(dh, dropped_H)

        # slicing layout different from cntk's implementation
        it_proj  = slice(proj4, stack_axis, 0 * stacked_dim, 1 * stacked_dim)  # split along stack_axis
        ft_proj  = slice(proj4, stack_axis, 1 * stacked_dim, 2 * stacked_dim)
        bit_proj = slice(proj4, stack_axis, 2 * stacked_dim, 3 * stacked_dim)  # g gate
        ot_proj  = slice(proj4, stack_axis, 3 * stacked_dim, 4 * stacked_dim)

        it = sigmoid(it_proj)                        # input gate(t)
        bit = it * activation(bit_proj)              # applied to tanh of input network

        ft = sigmoid(ft_proj)                        # forget-me-not gate(t)
        bft = ft * dc                                # applied to cell(t-1)

        ct = bft + bit                               # c(t) is sum of both

        ot = sigmoid(ot_proj)                        # output gate(t)
        ht = ot * activation(ct)                     # applied to tanh(cell(t))
        return ht, ct

    return lstm
