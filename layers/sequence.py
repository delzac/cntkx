'''
First / higher-order functions over sequences, like :func:`Recurrence`.
'''


import cntk as C
import cntkx as Cx
from cntk.ops import splice
from cntk.layers import SentinelValueForAutoSelectRandomSeed
from cntk.layers.blocks import _get_initial_state_or_default, _inject_name
from cntk.default_options import get_default_override, default_override_or
from cntk.layers.sequence import _santize_step_function, RecurrenceFrom
from .layers import Embedding


# TODO: Include RecurrenceFrom in here
def Recurrence(step_function, go_backwards=default_override_or(False), initial_state=default_override_or(0),
               return_full_state=False, dropout_rate_input=None,
               dropout_rate_output=None, seed=SentinelValueForAutoSelectRandomSeed, name=''):
    '''
    Recurrence(step_function, go_backwards=False, initial_state=0, return_full_state=False, name='')

    Recurrence has option to variationally dropout input and output.

    Layer factory function that implements a recurrent model, including the common RNN, LSTM, and GRU recurrences.
    This factory function creates a function that runs a step function recurrently over an input sequence,
    where in each step, Recurrence() will pass to the step function a data input as well as the output of the
    previous step.
    The following pseudo-code repesents what happens when you call a `Recurrence()` layer::

      # pseudo-code for y = Recurrence(step_function)(x)
      #  x: input sequence of tensors along the dynamic axis
      #  y: resulting sequence of outputs along the same dynamic axis
      y = []              # result sequence goes here
      s = initial_state   # s = output of previous step ("state")
      for x_n in x:       # pseudo-code for looping over all steps of input sequence along its dynamic axis
          s = step_function(s, x_n)  # pass previous state and new data to step_function -> new state
          y.append(s)

    The common step functions are :func:`~cntk.layers.blocks.LSTM`, :func:`~cntk.layers.blocks.GRU`, and :func:`~cntk.layers.blocks.RNNStep`,
    but the step function can be any :class:`~cntk.ops.functions.Function` or Python function.
    The signature of a step function with a single state variable must be
    ``(h_prev, x) -> h``, where ``h_prev`` is the previous state, ``x`` is the new
    data input, and the output is the new state.
    The step function will be called item by item, resulting in a sequence of the same length as the input.

    Step functions can have more than one state output, e.g. :func:`~cntk.layers.blocks.LSTM`.
    In this case, the first N arguments are the previous state, followed by one more argument that
    is the data input; and its output must be a tuple of N values.
    In this case, the recurrence operation will, by default, return the first of the state variables
    (in the LSTM case, the ``h``), while additional state variables are internal (like the LSTM's ``c``).
    If all state variables should be returned, pass ``return_full_state=True``.

    To provide your own step function, just use any :class:`~cntk.ops.functions.Function` (or equivalent Python function) that
    has a signature as described above.
    For example, a cumulative sum over a sequence can be computed as ``Recurrence(plus)``,
    where each step consists of `plus(s,x_n)`, where `s` is the output of the previous call
    and hence the cumulative sum of all elements up to `x_n`.
    Another example is a GRU layer with projection, which could be realized as ``Recurrence(GRU(500) >> Dense(200))``,
    where the projection is applied to the hidden state as fed back to the next step.
    ``F>>G`` is a short-hand for ``Sequential([F, G])``.

    Optionally, the recurrence can run backwards. This is useful for constructing bidirectional models.

    ``initial_state`` must be a constant. To pass initial_state as a data input, e.g. for a sequence-to-sequence
    model, use :func:`~cntk.layers.sequence.RecurrenceFrom()` instead.

    Note: ``Recurrence()`` is the equivalent to what in functional programming is often called ``scanl()``.

    Example:
     >>> from cntk.layers import Sequential
     >>> from cntk.layers.typing import Tensor, Sequence

     >>> # a recurrent LSTM layer
     >>> lstm_layer = Recurrence(LSTM(500))

     >>> # a bidirectional LSTM layer
     >>> # using function tuples to implement a bidirectional LSTM
     >>> bi_lstm_layer = Sequential([(Recurrence(LSTM(250)),                      # first tuple entry: forward pass
     ...                              Recurrence(LSTM(250), go_backwards=True)),  # second: backward pass
     ...                             splice])                                     # splice both on top of each other
     >>> bi_lstm_layer.update_signature(Sequence[Tensor[13]])
     >>> bi_lstm_layer.shape   # shape reflects concatenation of both output states
     (500,)
     >>> tuple(str(axis.name) for axis in bi_lstm_layer.dynamic_axes)  # (note: str() needed only for Python 2.7)
     ('defaultBatchAxis', 'defaultDynamicAxis')

     >>> # custom step function example: using Recurrence() to
     >>> # compute the cumulative sum over an input sequence
     >>> x = C.input_variable(**Sequence[Tensor[2]])
     >>> x0 = np.array([[   3,    2],
     ...                [  13,   42],
     ...                [-100, +100]])
     >>> cum_sum = Recurrence(C.plus, initial_state=Constant([0, 0.5]))
     >>> y = cum_sum(x)
     >>> y(x0)
     [array([[   3. ,    2.5],
             [  16. ,   44.5],
             [ -84. ,  144.5]], dtype=float32)]

    Args:
     step_function (:class:`~cntk.ops.functions.Function` or equivalent Python function):
      This function must have N+1 inputs and N outputs, where N is the number of state variables
      (typically 1 for GRU and plain RNNs, and 2 for LSTMs).
     go_backwards (bool, defaults to ``False``): if ``True`` then run the recurrence from the end of the sequence to the start.
     initial_state (scalar or tensor without batch dimension; or a tuple thereof):
      the initial value for the state. This can be a constant or a learnable parameter.
      In the latter case, if the step function has more than 1 state variable,
      this parameter must be a tuple providing one initial state for every state variable.
     return_full_state (bool, defaults to ``False``): if ``True`` and the step function has more than one
      state variable, then the layer returns a all state variables (a tuple of sequences);
      whereas if not given or ``False``, only the first state variable is returned to the caller.
     dropout_rate_input (float): dropout for input
     dropout_rate_output (float): dropout for output
     seed (int): seed for randomisation
     name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`:
        A function that accepts one argument (which must be a sequence) and performs the recurrent operation on it
    '''

    # BUGBUG: the cum_sum expression in the docstring should be this:
    # cum_sum = Recurrence(C.plus, initial_state=np.array([0, 0.5]))
    # BUGBUG: whereas passing a NumPy array fails with "TypeError: cannot convert value of dictionary"
    # cum_sum = Recurrence(C.plus, initial_state=Constant([0, 0.5]))

    go_backwards  = get_default_override(Recurrence, go_backwards=go_backwards)
    initial_state = get_default_override(Recurrence, initial_state=initial_state)
    initial_state = _get_initial_state_or_default(initial_state)

    step_function = _santize_step_function(step_function)

    dropout_input = None
    if dropout_rate_input:
        dropout_input = VariationalDropout(dropout_rate=dropout_rate_input, seed=seed, name='variational_dropout_input')

    dropout_output = None
    if dropout_rate_output:
        dropout_output = VariationalDropout(dropout_rate=dropout_rate_output, seed=seed, name='variational_dropout_output')

    # get signature of step function
    #*prev_state_args, _ = step_function.signature  # Python 3
    prev_state_args = step_function.signature[0:-1]

    if len(step_function.outputs) != len(prev_state_args):
        raise TypeError('Recurrence: number of state variables inconsistent between create_placeholder() and recurrent block')

    # initial state can be a single value or one per state variable (if more than one, like for LSTM)
    if isinstance(initial_state, tuple) and len(initial_state) == 1:
        initial_state = initial_state[0]
    if not isinstance(initial_state, tuple):
        # TODO: if initial_state is a CNTK Function rather than an initializer, then require to pass it multiple times; otherwise broadcast to all
        initial_state = tuple(initial_state for out_var in prev_state_args)

    # express it w.r.t. RecurrenceFrom
    recurrence_from = RecurrenceFrom(step_function, go_backwards, return_full_state) # :: (x, state seq) -> (new state seq)

    # function that this layer represents
    @C.Function
    def recurrence(x):
        dropped_x = dropout_input(x) if dropout_input else x
        y = recurrence_from(*(initial_state + (dropped_x,)))
        dropped_y = dropout_output(y) if dropout_output else y
        return dropped_y

    return _inject_name(recurrence, name)


def BiRecurrence(step_function: C.Function, initial_state=0, dropout_rate_input=None, dropout_rate_output=None,
                 weight_tie: bool = False, seed=SentinelValueForAutoSelectRandomSeed, name=''):
    """ Wrapper to create a bidirectional rnn

    Also comes with the option to to half the number of parameters required by  bidirectional recurrent layer.
    This is done by only using one recurrent unit to do both forward and backward computation instead of
    the usual two. A forward and backward token is used to initialise the hidden state so that the recurrent
    unit can tell the directionality.

    More details can be found in the paper 'Efficient Bidirectional Neural Machine Translation' (https://arxiv.org/abs/1908.09329)

    Example:
        a = C.sequence.input_variable(10)
        b = BiRecurrence(LSTM(100), weight_tie=True)(a)

        assert b.shape == (200, )

    Arguments:
        step_function (:class:`~cntk.ops.functions.Function` or equivalent Python function):
            This function must have N+1 inputs and N outputs, where N is the number of state variables
            (typically 1 for GRU and plain RNNs, and 2 for LSTMs).
        initial_state:
        dropout_rate_input: variational dropout on input
        dropout_rate_output: variational dropoput on output
        weight_tie (bool): whether to use only one recurrent function for computation in both direction.
        seed (int): seed for randomisation
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`:
        A function that accepts one argument (which must be a sequence) and performs the recurrent operation on it
    """
    fxn1 = step_function
    fxn2 = step_function.clone(C.CloneMethod.clone, {}) if not weight_tie else fxn1

    forward_token = initial_state
    backward_token = initial_state
    if weight_tie:
        forward_token = C.Parameter(shape=(-1,), init=C.glorot_normal(), name='f_token')
        backward_token = C.Parameter(shape=(-1,), init=C.glorot_normal(), name='b_token')

    forward = Recurrence(fxn1, dropout_rate_input=dropout_rate_input, dropout_rate_output=dropout_rate_output, initial_state=forward_token, seed=seed)
    backward = Recurrence(fxn2, dropout_rate_input=dropout_rate_input, dropout_rate_output=dropout_rate_output, initial_state=backward_token, seed=seed, go_backwards=True)

    @C.Function
    def inner(x):
        output = C.splice(forward(x), backward(x), axis=-1)
        return C.layers.Label(name)(output) if name else output

    return inner


def PyramidalBiRecurrence(step_fxn_f, step_fxn_b, width, initial_state_f=0, initial_state_b=0,
                           variational_dropout_rate_input=None, variational_dropout_rate_output=None,
                           seed=SentinelValueForAutoSelectRandomSeed, name=''):
    """ Pyramidal bidirectional recurrence

    Implements a bi-directional recurrence of the step functions given with a pyramidal structure.

    PyramidalBiRecurrence reduces the sequence length and simultaneously increase dimension by width factor.
    Its typically used in acoustic model to keep runtime manageable.

    For mode details, please refer to "Listen, attend and spell" by Chan et al. (https://arxiv.org/abs/1508.01211)

          concatenated
          [f1,b8, f2, b7]
           /    \
          /      \
    <--- b8 <--- b7 <--- b6 <--- b5 <--- b4 <--- b3 <--- b2 <--- b1
    ---> f1 ---> f2 ---> f3 ---> f4 ---> f5 ---> f6 ---> f7 ---> f8

    Example:
        import cntk as C
        import cntkx as Cx

        a = C.sequence.input_variable(10)
        b = Cx.layers.PyramidalBiRecurrence(LSTM(30), LSTM(30), width=2)(a)

        assert b.shape == (30 * 2 * 2, )  # with sequence length also reduced by width factor

    Arguments:
        step_fxn_f: step function in the forward direction
        step_fxn_b: step function in the backward direction
        width (int): width of window in pyramidal structure
        initial_state_f (scalar or tensor without batch dimension; or a tuple thereof):
            the initial value for the state. This can be a constant or a learnable parameter.
            In the latter case, if the step function has more than 1 state variable,
            this parameter must be a tuple providing one initial state for every state variable.
        initial_state_b (scalar or tensor without batch dimension; or a tuple thereof):
            the initial value for the state. This can be a constant or a learnable parameter.
            In the latter case, if the step function has more than 1 state variable,
            this parameter must be a tuple providing one initial state for every state variable.
        variational_dropout_rate_input (float): dropout for input
        variational_dropout_rate_output (float): dropout for output
        seed (int): seed for randomisation
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    forward = Recurrence(step_fxn_f, initial_state=initial_state_f, go_backwards=False,
                         return_full_state=False, dropout_rate_input=variational_dropout_rate_input,
                         dropout_rate_output=variational_dropout_rate_output, seed=seed, name='Pyramidal_Forward')

    backward = Recurrence(step_fxn_b, initial_state=initial_state_b, go_backwards=True,
                          return_full_state=False, dropout_rate_input=variational_dropout_rate_input,
                          dropout_rate_output=variational_dropout_rate_output, seed=seed, name='Pyramidal_Backward')

    @C.Function
    def inner(x):
        y = C.splice(forward(x), backward(x), axis=-1)
        z = Cx.sequence.window(y, width)
        return z

    return inner


def VariationalDropout(dropout_rate: float, seed=SentinelValueForAutoSelectRandomSeed, name=''):
    """ Variational dropout uses the same dropout mask at each time step (i.e. across the dynamic sequence axis)

    Example:
        a = C.sequence.input_variable(10)
        b = VariationalDropout(0.1)(a)

        assert b.shape == a.shape

    Arguments:
        dropout_rate (float): probability of dropping out an element
        seed (int): seed for randomisation
        name (str, defaults to ''): the name of the Function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the operation to it

    """
    dropout = C.layers.Dropout(dropout_rate, seed=seed)

    @C.BlockFunction('VariationalDropout', name)
    def inner(x):
        mask = dropout(C.ones_like(C.sequence.first(x)))
        mask = C.sequence.broadcast_as(mask, x)
        return mask * x

    return inner
