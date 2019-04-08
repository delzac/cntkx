import cntk as C
from cntkx.layers import QRNN, Recurrence, IndyLSTM, IndRNN, WeightDroppedLSTM
from cntk.layers import Dense, LSTM
import numpy as np
import random
import time
from pprint import pprint
from operator import mul
from functools import reduce


def generate_variable_10(nb_samples=1000, dim=3):
    """
    Generate a dataset of sequences where sequence length is between 5 to 20. Sequence will have values 1 or 0.
    Target will be the sum of the sequence.
    """
    r = [random.randint(5, 20) for __ in range(nb_samples)]
    x = [np.random.randint(2, size=(t, dim)).astype(np.float32) for t in r]
    y = np.array([arr.sum() for arr in x])[..., None]
    return x, y


def model_qrnn(input_tensor, hidden_dim):
    hidden = QRNN(window=2, hidden_dim=hidden_dim)(input_tensor)
    prediction = Dense(1)(C.sequence.last(hidden))
    return prediction


def model_lstm(input_tensor, hidden_dim):
    hidden = Recurrence(LSTM(hidden_dim))(input_tensor)
    prediction = Dense(1)(C.sequence.last(hidden))
    return prediction


def model_wdlstm(input_tensor, hidden_dim, dropout):
    hidden = Recurrence(WeightDroppedLSTM(hidden_dim, dropout))(input_tensor)
    prediction = Dense(1)(C.sequence.last(hidden))
    return prediction

def model_indy_lstm(input_tensor, hidden_dim):
    hidden = Recurrence(IndyLSTM(hidden_dim))(input_tensor)
    prediction = Dense(1)(C.sequence.last(hidden))
    return prediction


def model_ind_rnn(input_tensor, hidden_dim):
    hidden = Recurrence(IndRNN(hidden_dim, C.relu))(input_tensor)
    prediction = Dense(1)(C.sequence.last(hidden))
    return prediction


n_epoch = 50
minibatch_size = 30

input_dim = 2
x, y = generate_variable_10(nb_samples=1000, dim=input_dim)
test_x, test_y = generate_variable_10(nb_samples=100, dim=input_dim)

input_tensor = C.sequence.input_variable(input_dim)
target_tensor = C.input_variable(1)

p_lstm = model_lstm(input_tensor, hidden_dim=100)
p_wdlstm = model_wdlstm(input_tensor, hidden_dim=100, dropout=0.1)
p_qrnn = model_qrnn(input_tensor, hidden_dim=800)
p_indy_lstm = model_indy_lstm(input_tensor, hidden_dim=200)
p_ind_rnn = model_ind_rnn(input_tensor, hidden_dim=600)

predictions = [p_lstm, p_wdlstm, p_qrnn, p_indy_lstm, p_ind_rnn]
block_names = ['lstm', 'wdlstm', 'qrnn', 'IndyLSTM', 'IndRNN']

performance = []
for block_name, prediction in zip(block_names, predictions):
    loss = C.squared_error(prediction, target_tensor)
    adam = C.adam(prediction.parameters, 0.1, 0.912)
    trainer = C.Trainer(prediction, (loss,), [adam])

    num_parameters = sum([reduce(mul, p.shape + (1,)) for p in prediction.parameters])

    history = []
    start = time.time()

    for epoch in range(n_epoch):

        for i in range(0, len(x), minibatch_size):
            lbound, ubound = i, i + minibatch_size
            x_mini = x[lbound:ubound]
            y_mini = y[lbound:ubound]
            trainer.train_minibatch({input_tensor: x_mini,
                                     target_tensor: y_mini})

            history.append(trainer.previous_minibatch_loss_average)
            # print(f"loss: {history[-1]}")

    duration = time.time() - start
    print(f"{block_name} training completed in {duration}s")

    loss_result = sum(loss.eval({input_tensor: test_x, target_tensor: test_y}))

    block_performance = (block_name, duration, loss_result, num_parameters)
    performance.append(block_performance)

for block_name, duration, loss_result, num_parameters in performance:
    print(f"name: {block_name}, duration: {duration}, loss: {loss_result}, parameter_count: {num_parameters}")