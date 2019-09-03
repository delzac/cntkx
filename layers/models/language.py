from ...layers import Recurrence, LSTM, Embedding
from cntk.layers import Label


def PretrainedWikitext103LanguageModel(model_file_path: str, weight_drop_rate: float = None, v_dropout_rate: float = None):
    """ General Language Model from fastai's ULMFIT by Jeremy Howard and Sebastian Ruder

    Universal  Language  ModelFine-tuning (ULMFiT) is an effective transfer learning
    method that can be applied to any task in NLP, and introduce techniques that are key
    for fine-tuning a language model.

    The paper 'Universal Language Model Fine-tuning for Text Classification' can be
    found at https://arxiv.org/abs/1801.06146

    This model is designed for use with parameters from 'fwd_wt103.h5'. The original pytorch model
    must be converted into a proper hdf5 file first before it can used with this model.

    url to download the original pytorch model and vocabulary/token list can be found here:
        http://files.fast.ai/models/wt103/

    The converted hdf5 file that can be used immediately with this model can be downloaded from the url below:
        https://1drv.ms/u/s!AjJ4XyC3prp8mItNxiawGK4gD8iMhA?e=wh7PLB

    Alternatively, you can download the original pytorch model and convert it using the
    'convert_pytorch_state_dict_to_h5_file' helper function found in cntkx.misc module.

    Example:
        vocab_size = 238462
        converted_hdf5_model_file_path = ''  # this is not the original pytorch model
        lm = PretrainedWikitext103LanguageModel(converted_hdf5_model_file_path)

        a = C.sequence.input_variable(vocab_size)
        prediction = lm(a)  # next-word-prediction
        features = prediction.features  # features of tokens

        assert prediction.shape == (vocab_size, )
        assert features.shape == (400, )

    Arguments:
        model_file_path (str): file path to the converted model (not the original pytorch model).
        weight_drop_rate (float): amount of weight drop to be done on the recurrent weights of the LSTM
        v_dropout_rate (float): amount of variational dropout to apply to input and outputs of the recurrent layers.

    Returns:
        :class:`~cntk.ops.functions.Function`:

    """
    import h5py
    model_params = h5py.File(model_file_path, 'r')
    layer_names = ['0.encoder.weight',
                   '0.encoder_with_dropout.embed.weight',
                   '0.rnns.0.module.bias_hh_l0',
                   '0.rnns.0.module.bias_ih_l0',
                   '0.rnns.0.module.weight_hh_l0_raw',
                   '0.rnns.0.module.weight_ih_l0',
                   '0.rnns.1.module.bias_hh_l0',
                   '0.rnns.1.module.bias_ih_l0',
                   '0.rnns.1.module.weight_hh_l0_raw',
                   '0.rnns.1.module.weight_ih_l0',
                   '0.rnns.2.module.bias_hh_l0',
                   '0.rnns.2.module.bias_ih_l0',
                   '0.rnns.2.module.weight_hh_l0_raw',
                   '0.rnns.2.module.weight_ih_l0',
                   '1.decoder.weight']

    hidden_dim0 = model_params['0.rnns.0.module.weight_ih_l0'].shape[0] // 4
    hidden_dim1 = model_params['0.rnns.1.module.weight_ih_l0'].shape[0] // 4
    hidden_dim2 = model_params['0.rnns.2.module.weight_ih_l0'].shape[0] // 4

    assert hidden_dim0 == 1150
    assert hidden_dim1 == 1150
    assert hidden_dim2 == 400

    embedding, predict = Embedding(shape=(), init=model_params['0.encoder.weight'][:], enable_weight_tying=True)

    rnn0 = LSTM(shape=(hidden_dim0,), weight_drop_rate=weight_drop_rate,
                ih_init=model_params['0.rnns.0.module.weight_ih_l0'][:].T,
                ih_bias=model_params['0.rnns.0.module.bias_ih_l0'][:],
                hh_init=model_params['0.rnns.0.module.weight_hh_l0_raw'][:].T,
                hh_bias=model_params['0.rnns.0.module.bias_hh_l0'][:],
                name='rnn0')
    rnn1 = LSTM(shape=(hidden_dim1,), weight_drop_rate=weight_drop_rate,
                ih_init=model_params['0.rnns.1.module.weight_ih_l0'][:].T,
                ih_bias=model_params['0.rnns.1.module.bias_ih_l0'][:],
                hh_init=model_params['0.rnns.1.module.weight_hh_l0_raw'][:].T,
                hh_bias=model_params['0.rnns.1.module.bias_hh_l0'][:],
                name='rnn1')
    rnn2 = LSTM(shape=(hidden_dim2,), weight_drop_rate=weight_drop_rate,
                ih_init=model_params['0.rnns.2.module.weight_ih_l0'][:].T,
                ih_bias=model_params['0.rnns.2.module.bias_ih_l0'][:],
                hh_init=model_params['0.rnns.2.module.weight_hh_l0_raw'][:].T,
                hh_bias=model_params['0.rnns.2.module.bias_hh_l0'][:],
                name='rnn2')

    def model(x):
        hidden = embedding(x)
        hidden = Recurrence(rnn0, dropout_rate_input=v_dropout_rate, dropout_rate_output=v_dropout_rate)(hidden)
        hidden = Recurrence(rnn1, dropout_rate_input=v_dropout_rate, dropout_rate_output=v_dropout_rate)(hidden)
        hidden = Recurrence(rnn2, dropout_rate_input=v_dropout_rate)(hidden)
        hidden = Label('features')(hidden)
        prediction = predict(hidden)
        return prediction

    return model
