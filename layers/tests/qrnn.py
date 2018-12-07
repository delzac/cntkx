import cntk as C
from cntkx.layers import QRNN
from cntk.layers import Dense, Recurrence, LSTM
import numpy as np
import random


def generate_variable_10(nb_samples=1000, dim=3):
    """
    Generate a dataset of sequences where sequence length is between 5 to 20. Sequence will have values 1 or 0.
    Target will be the sum of the sequence.
    """
    r = [random.randint(5, 20) for __ in range(nb_samples)]
    x = [np.random.randint(2, size=(t, dim)).astype(np.float32) for t in r]
    y = np.array([arr.sum() for arr in x])[..., None]
    return x, y


input_dim = 2
x, y = generate_variable_10(nb_samples=1000, dim=input_dim)

hidden_dim = 100

input_tensor = C.sequence.input_variable(input_dim)
target_tensor = C.input_variable(1)

qrnn = QRNN(hidden_dim=hidden_dim)(input_tensor)
# rnn = Recurrence(LSTM(hidden_dim))(input_tensor)
prediction = Dense(1)(C.sequence.last(qrnn))

loss = C.squared_error(prediction, target_tensor)
sgd_m = C.momentum_sgd(prediction.parameters, 0.01, 0.912)
trainer = C.Trainer(prediction, (loss,), [sgd_m])

n_epoch = 0
minibatch_size = 30

for epoch in range(n_epoch):

    for i in range(0, len(x), minibatch_size):
        lbound, ubound = i, i + minibatch_size
        x_mini = x[lbound:ubound]
        y_mini = y[lbound:ubound]
        trainer.train_minibatch({input_tensor: x_mini,
                                 target_tensor: y_mini})

        print(f"loss: {trainer.previous_minibatch_loss_average}")

# TODO: pad sequence check google
n = np.random.randint(2, size=(6, input_dim)).astype(np.float32)
print(prediction.eval({input_tensor: [n]}))
print(n.sum())
