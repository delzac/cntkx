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
from cntk.ops import times, slice, sigmoid, tanh
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
        'RNNStep': 1,
        'GRU': 3,
        'WeightDroppedLSTM': 4
    }[type]
    cell_shape_stacked = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times
    cell_shape_list[stack_axis] = stacked_dim * {
        'RNNStep': 1,
        'GRU': 2,
        'WeightDroppedLSTM': 4
    }[type]
    cell_shape_stacked_H = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times

    # parameters
    b  = Parameter(            cell_shape_stacked,   init=init_bias, name='b')                              # bias
    W  = Parameter(_INFERRED + cell_shape_stacked,   init=init,      name='W')                              # input
    H  = Parameter(shape     + cell_shape_stacked_H, init=init,      name='H')                              # hidden-to-hidden
    H1 = Parameter(shape     + cell_shape,           init=init,      name='H1') if type == 'GRU' else None  # hidden-to-hidden
    Ci = Parameter(            cell_shape,           init=init,      name='Ci') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
    Cf = Parameter(            cell_shape,           init=init,      name='Cf') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
    Co = Parameter(            cell_shape,           init=init,      name='Co') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}

    Wmr = Parameter(cell_shape + shape, init=init, name='P') if has_projection else None  # final projection

    # each use of a stabilizer layer must get its own instance
    Sdh = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='dh_stabilizer')
    Sdc = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='dc_stabilizer')
    Sct = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='c_stabilizer')
    Sht = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='P_stabilizer')

    # DropConnecet
    droppout = C.layers.Dropout(dropout_rate=dropout_rate, seed=seed, name='h_dropout')

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
        proj4 = b + times(x, W) + times(dhs, droppout(H))

        it_proj  = slice (proj4, stack_axis, 0*stacked_dim, 1*stacked_dim)  # split along stack_axis
        bit_proj = slice (proj4, stack_axis, 1*stacked_dim, 2*stacked_dim)
        ft_proj  = slice (proj4, stack_axis, 2*stacked_dim, 3*stacked_dim)
        ot_proj  = slice (proj4, stack_axis, 3*stacked_dim, 4*stacked_dim)

        # helper to inject peephole connection if requested
        def peep(x, c, C):
            return x + C * c if use_peepholes else x

        it = sigmoid (peep (it_proj, dcs, Ci))        # input gate(t)
        # TODO: should both activations be replaced?
        bit = it * activation (bit_proj)              # applied to tanh of input network

        ft = sigmoid (peep (ft_proj, dcs, Cf))        # forget-me-not gate(t)
        bft = ft * dc                                 # applied to cell(t-1)

        ct = bft + bit                                # c(t) is sum of both

        ot = sigmoid (peep (ot_proj, Sct(ct), Co))    # output gate(t)
        ht = ot * activation (ct)                     # applied to tanh(cell(t))

        c = ct                                        # cell value
        h = times(Sht(ht), Wmr) if has_projection else \
            ht

        # returns the new state as a tuple with names but order matters
        #return (Function.NamedOutput(h=h), Function.NamedOutput(c=c))
        return h, c

    # GRU model function
    # in this case:
    #   (dh, x) --> (h)
    # e.g. https://en.wikipedia.org/wiki/Gated_recurrent_unit
    def gru(dh, x):

        dhs = Sdh(dh)  # previous value, stabilized
        # note: input does not get a stabilizer here, user is meant to do that outside

        # projected contribution from input(s), hidden, and bias
        projx3 = b + times(x, W)
        projh2  = times(dhs, H)

        zt_proj = slice (projx3, stack_axis, 0*stacked_dim, 1*stacked_dim) + slice (projh2, stack_axis, 0*stacked_dim, 1*stacked_dim)
        rt_proj = slice (projx3, stack_axis, 1*stacked_dim, 2*stacked_dim) + slice (projh2, stack_axis, 1*stacked_dim, 2*stacked_dim)
        ct_proj = slice (projx3, stack_axis, 2*stacked_dim, 3*stacked_dim)

        zt = sigmoid (zt_proj)        # update gate z(t)

        rt = sigmoid (rt_proj)        # reset gate r(t)

        rs = dhs * rt        # "cell" c
        ct = activation (ct_proj + times(rs, H1))

        ht = (1 - zt) * ct + zt * dhs # hidden state ht / output

        # for comparison: CUDNN_GRU
        # i(t) = sigmoid(W_i x(t) +          R_i h(t-1)  + b_Wi + b_Ru)
        # r(t) = sigmoid(W_r x(t) +          R_r h(t-1)  + b_Wr + b_Rr)   --same up to here
        # h'(t) =   tanh(W_h x(t) + r(t) .* (R_h h(t-1)) + b_Wh + b_Rh)   --r applied after projection? Would make life easier!
        # h(t) = (1 - i(t) .* h'(t)) + i(t) .* h(t-1)                     --TODO: need to confirm bracketing with NVIDIA

        h = times(Sht(ht), Wmr) if has_projection else \
            ht

        # returns the new state as a tuple with names but order matters
        #return Function.NamedOutput(h=h)
        return h

    def rnn_step(dh, x):
        dhs = Sdh(dh)  # previous value, stabilized
        ht = activation (times(x, W) + times(dhs, H) + b)
        h = times(Sht(ht), Wmr) if has_projection else \
            ht
        #return Function.NamedOutput(h=h)
        return h

    function = {
        'RNNStep': rnn_step,
        'GRU':     gru,
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
     >>> from cntk.layers import *
     >>> lstm_layer = Recurrence(LSTM(500))

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


def RNNStep(shape, cell_shape=None, activation=default_override_or(sigmoid),
            init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
            enable_self_stabilization=default_override_or(False),
            name=''):
    '''
    RNNStep(shape, cell_shape=None, activation=sigmoid, init=glorot_uniform(), init_bias=0, enable_self_stabilization=False, name='')

    Layer factory function to create a plain RNN block for use inside a recurrence.
    The RNN block implements one step of the recurrence and is stateless. It accepts the previous state as its first argument,
    and outputs its new state.

    Example:
     >>> # a plain relu RNN layer
     >>> from cntk.layers import *
     >>> relu_rnn_layer = Recurrence(RNNStep(500, activation=C.relu))

    Args:
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        cell_shape (tuple, defaults to `None`): if given, then the output state is first computed at `cell_shape`
         and linearly projected to `shape`
        activation (:class:`~cntk.ops.functions.Function`, defaults to signmoid): function to apply at the end, e.g. `relu`
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to `glorot_uniform`): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        enable_self_stabilization (bool, defaults to `False`): if `True` then add a :func:`~cntk.layers.blocks.Stabilizer`
         to all state-related projections (but not the data input)
        name (str, defaults to ''): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`:
        A function ``(prev_h, input) -> h`` where ``h = activation(input @ W + prev_h @ R + b)``
    '''

    activation                = get_default_override(RNNStep, activation=activation)
    init                      = get_default_override(RNNStep, init=init)
    init_bias                 = get_default_override(RNNStep, init_bias=init_bias)
    enable_self_stabilization = get_default_override(RNNStep, enable_self_stabilization=enable_self_stabilization)

    return _RecurrentBlock('RNNStep', shape, cell_shape, activation=activation, use_peepholes=False,
                           init=init, init_bias=init_bias,
                           enable_self_stabilization=enable_self_stabilization, name=name)


def GRU(shape, cell_shape=None, activation=default_override_or(tanh),
        init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
        enable_self_stabilization=default_override_or(False),
        name=''):
    '''
    GRU(shape, cell_shape=None, activation=tanh, init=glorot_uniform(), init_bias=0, enable_self_stabilization=False, name='')

    Layer factory function to create a GRU block for use inside a recurrence.
    The GRU block implements one step of the recurrence and is stateless. It accepts the previous state as its first argument,
    and outputs its new state.

    Example:
     >>> # a gated recurrent layer
     >>> from cntk.layers import *
     >>> gru_layer = Recurrence(GRU(500))

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
        A function ``(prev_h, input) -> h`` that implements one step of a recurrent GRU layer.
    '''

    activation                = get_default_override(GRU, activation=activation)
    init                      = get_default_override(GRU, init=init)
    init_bias                 = get_default_override(GRU, init_bias=init_bias)
    enable_self_stabilization = get_default_override(GRU, enable_self_stabilization=enable_self_stabilization)

    return _RecurrentBlock('GRU', shape, cell_shape, activation=activation, use_peepholes=False,
                           init=init, init_bias=init_bias,
                           enable_self_stabilization=enable_self_stabilization, name=name)
