import cntk as C
from cntk.layers import Dense
import numpy as np
from cntkx.learners import RAdam, adam_exponential_warmup_schedule


def test_RAdam():
    beta2 = 0.999
    a = C.input_variable(shape=(1,))
    c = Dense(shape=(1,))(a)

    z = adam_exponential_warmup_schedule(1, beta2)

    loss = C.squared_error(a, c)
    adam = RAdam(c.parameters, 1, 0.912, beta2=beta2, epoch_size=3)
    trainer = C.Trainer(c, (loss, ), [adam])

    n = np.random.random((3, 1))
    for i in range(10_000):
        assert z[i] == adam.learning_rate()
        # print(f"iter: {i}, lr: {adam.learning_rate()}")
        trainer.train_minibatch({a: n})
