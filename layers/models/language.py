from ...layers import Recurrence, LSTM, Embedding
import h5py


def PretrainedWikitext103LanguageModel(model_file_path: str, embed_dropout_rate: float = None,
                                       weight_drop_rate: float = None, v_dropout_rate: float = None):
    """ Model from fastai """
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

    embedding = Embedding(shape=(), init=model_params['0.encoder.weight'][:])

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

    # TODO: Check if weight is tied to encoding embedding
    predict = Embedding(shape=(), init=model_params['1.decoder.weight'][:].T)

    def model(x):
        hidden = embedding(x)
        hidden = Recurrence(rnn0, dropout_rate_input=v_dropout_rate, dropout_rate_output=v_dropout_rate)(hidden)
        hidden = Recurrence(rnn1, dropout_rate_input=v_dropout_rate, dropout_rate_output=v_dropout_rate)(hidden)
        hidden = Recurrence(rnn2, dropout_rate_input=v_dropout_rate)(hidden)
        prediction = predict(hidden)
        return prediction

    return model
