import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import cntk as C
from cntk.layers import Dense
from cntkx.losses import gaussian_mdn_loss
from cntkx.ops import gaussian_mdn_coeff, sample_gaussian_mdn


def generate_sine_data(nb_samples: int):
    """ Generates a sine wave dataset """
    e = 2 * np.random.random(size=(nb_samples,)) / 10 - 0.1
    x = np.arange(nb_samples) / nb_samples
    y = x + 0.3 * np.sin(2 * np.pi * x) + e
    print("shape of x and y: {0} {1}".format(x.shape, y.shape))
    return x, y


def make_range(m, s):
    return m + s, m, m - s


# ========================================================================
#                        Configuration
# ========================================================================
ndim = 1
nmix = 3

# ========================================================================
# ========================================================================

x, y = generate_sine_data(1000)  # swap y and x to convert problem from one-to-many to many-to-one
x, y = x[:, None].astype(np.float32), y[:, None].astype(np.float32)
x, y = shuffle(x, y)

input_tensor = C.input_variable(1, name="input_tensor")
target_tensor = C.input_variable(1, name="target_tensor")

# model
inner = Dense(50, activation=C.relu)(input_tensor)
inner = Dense(50, activation=C.relu)(inner)
# inner = Dense(30, activation=C.relu)(inner)
prediction_tensor = Dense((ndim + 2) * nmix, activation=None)(inner)

sampled = sample_gaussian_mdn(prediction_tensor, nmix, ndim)

loss = gaussian_mdn_loss(prediction_tensor, target_tensor, nmix=nmix, ndim=ndim)
sgd_m = C.momentum_sgd(prediction_tensor.parameters, 0.004, 0.9, l2_regularization_weight=0.02)
trainer = C.Trainer(prediction_tensor, (loss, ), [sgd_m])

# training loop
num_epoch = 800
minibatch_size = 200

for epoch in range(num_epoch):
    x, y = shuffle(x, y)

    for i in range(0, x.shape[0], minibatch_size):
        lbound, ubound = i, i + minibatch_size
        x_mini = x[lbound:ubound]
        y_mini = y[lbound:ubound]
        trainer.train_minibatch({input_tensor: x_mini,
                                 target_tensor: y_mini})

        print(f"loss: {trainer.previous_minibatch_loss_average}")

prediction = prediction_tensor.eval({input_tensor: x})
coeff = C.combine(gaussian_mdn_coeff(prediction_tensor, nmix=nmix, ndim=ndim)).eval({input_tensor: x})

# calculate range of mdn
alpha, mu, sigma = coeff.values()
m = np.sum(alpha * np.squeeze(mu), axis=-1)
s = np.sum(alpha * np.squeeze(sigma), axis=-1)
p1, p2, p3 = make_range(m, s)

print(prediction.shape, p1.shape, p2.shape, x.shape)

results = sampled.eval({input_tensor: x})
results = np.squeeze(results)

plt.scatter(x, y, s=2, c='blue')
plt.scatter(x, p1, s=2, c='red')
plt.scatter(x, p2, s=2, c='red')
plt.scatter(x, p3, s=2, c='red')
plt.scatter(x, results, s=2, c='green')

plt.show()
