import cntk as C
from cntkx.learners import CyclicalLearningRate
import matplotlib.pyplot as plt


def test_clr():
    base_lr = 0.1
    max_lr = 1
    minibatch_size = 16
    step_size = 100

    a = C.input_variable(10)
    model = C.layers.Dense(10)(a)
    sgd = C.sgd(model.parameters, 0.01)
    clr = CyclicalLearningRate(sgd,
                               base_lr=base_lr,
                               max_lr=max_lr,
                               minibatch_size=minibatch_size,
                               ramp_up_step_size=2000,
                               ramp_down_step_size=4000,)
    lr_schedule = clr.get_lr_schedule()

    plt.scatter(range(lr_schedule.shape[0]), lr_schedule, s=1)
    plt.show()
