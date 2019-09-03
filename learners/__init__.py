import cntk as C
import math
import numpy as np
from typing import List


class CyclicalLearningRate(object):
    """
    Cyclical learning rate is an implementation to that  practically eliminates
    the need to experimentally find the best values and schedule  for  the
    global  learning  rates.

    Instead  of  monotonically decreasing the learning rate, this method lets the
    learning  rate  cyclically  vary  between  reasonable  boundary  values

    Training  with  cyclical  learning  rates  instead of  fixed  values  achieves
    improved  classification  accuracy without a need to tune and often in fewer iterations.

    This is an CNTK implementation of the following paper:
    Cyclical Learning Rates for Training Neural Networks by Leslie N. Smith: https://arxiv.org/abs/1506.01186

    The policy cycles the learning rate between two boundaries with a constant frequency, as detailed in
    the paper.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CyclicalLeaningRate`.

    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.

    This implementation was adapted from the github repo:
    `bckenstler/CLR` and 'anandsaha/pytorch.cyclic.learning.rate'


    Args:
        parameter_learner (learner): list of cntk learner
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        minibatch_size (int): Number of samples in one minibatch
        lr_policy (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular2'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma ** iterations
            Default: 0.99994
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            lr_policy is ignored
            Default: None
        scale_by (str): scale by either number of training iterations or training cycles.
            Only used if custom scaling policy scale_fn is defined.
        record_history (bool): If True, loss & learn rate will be recorded. get loss lr
            history functions can then be used.
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
     >>> model = C.layers.Dense(10)(C.input_variable(10))
     >>> sgd_momentum = C.momentum_sgd(model.parameters, 0.1, 0.9)
     >>> clr = CyclicalLeaningRate(sgd_momentum, minibatch_size=32)

     >>> for epoch in range(10):
     ...     for batch in range(100):
     ...         # trainer.train_minibatch(...)
     ...         clr.batch_step()  # must be called once for every training iteration/update
    """

    def __init__(self, parameter_learner, base_lr=1e-3, max_lr=6e-3, minibatch_size=None,
                 ramp_up_step_size=2000, ramp_down_step_size: int = None, lr_policy='triangular2', gamma=0.99994,
                 scale_fn=None, scale_by="cycle", record_history=False, last_batch_iteration=-1):

        if lr_policy not in ['triangular', 'triangular2', 'exp_range'] and scale_fn is None:
            raise ValueError('lr_policy is invalid and scale_fn is None')

        if scale_by not in ['iteration', 'cycle']:
            raise ValueError("Can only scale by iteration or cycle")

        self.parameter_learner = parameter_learner
        self.base_lr = base_lr
        self.max_lr = max_lr

        self.ramp_up_step_size = ramp_up_step_size
        self.ramp_down_step_size = ramp_down_step_size or ramp_up_step_size
        self.cycle_size = self.ramp_up_step_size + self.ramp_down_step_size
        self.minibatch_size = minibatch_size

        self.lr_policy = lr_policy
        self.gamma = gamma
        self.record_history = record_history

        if scale_fn is None:
            if self.lr_policy == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_by = 'cycle'
            elif self.lr_policy == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_by = 'cycle'
            elif self.lr_policy == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_by = 'iteration'
        else:
            self.scale_fn = scale_fn
            self.scale_by = scale_by

        self.loss = []
        self.lrs: List[float] = []  # [(learn1_lr, ), (learner1_lr, ), ... ]
        self.current_lr = 0
        self.last_batch_iteration = last_batch_iteration
        self.batch_step()

    def _triangular_scale_fn(self, x) -> float:
        return 1.

    def _triangular2_scale_fn(self, x) -> float:
        return 1 / (2. ** (x - 1))

    def slanted_triangle_scale_fn(self, x) -> float:
        return

    def _exp_range_scale_fn(self, x) -> float:
        return self.gamma ** x

    def batch_step(self, previous_minibatch_loss=None):
        """
        Updates learners with new learning rate after one training iteration is complete.
        Must be called once for every training iteration/update.
        """

        self.last_batch_iteration += 1
        lr = self.get_lr()
        self.current_lr = lr
        
        # loss and learn rate gets recorded in pre-training mode
        if self.record_history and previous_minibatch_loss:
            self.loss.append(previous_minibatch_loss)
            self.lrs.append(lr)

        self.parameter_learner.reset_learning_rate(C.learning_parameter_schedule(lr, minibatch_size=self.minibatch_size))
        return None

    def get_lr(self) -> float:
        """ Get learning rate based on last_batch_iteration count """
        # Cycle number
        cycle_num: int = math.floor(1 + self.last_batch_iteration / self.cycle_size)

        # number of batch steps made since last complete cycle
        iteration_since_last_cycle = self.last_batch_iteration % self.cycle_size

        # vary between the range of 0 and 1
        # start -> mid -> end
        # 1     ->  0  ->  1
        # proportion_of_step_completed = abs(self.last_batch_iteration / step_size - 2 * cycle_num + 1)

        # ramping up or down
        is_ramp_up = True if iteration_since_last_cycle <= self.ramp_up_step_size else False

        num_batch_steps_in_ramp = iteration_since_last_cycle
        if is_ramp_up:
            base_height = (self.max_lr - self.base_lr) * (num_batch_steps_in_ramp / self.ramp_up_step_size)
        else:  # ramp_down
            num_batch_steps_in_ramp = iteration_since_last_cycle - self.ramp_up_step_size
            base_height = (self.max_lr - self.base_lr) * (1 - num_batch_steps_in_ramp / self.ramp_down_step_size)

        # base_height = (self.max_lr - self.base_lr) * max(0., (1 - proportion_of_step_completed))
        xx = cycle_num if self.scale_by == "cycle" else self.last_batch_iteration
        lr = self.base_lr + base_height * self.scale_fn(xx)
        return lr

    def get_lr_schedule(self, number_of_cycles=4) -> np.ndarray:
        """ returns how the learn rate schedule will be like. Useful to check if your
        custom learning policy is working as intended.

        The returned lr_schedule can be visualised in the following:
            import matplotlib.pyplot as plt

            plt.scatter(range(lr_schedule.shape[0]), lr_schedule, s=1)
            plt.show()

        Arguments:
            number_of_cycles (int):

        Returns:
            np.ndarray 1-d

        """
        store_last_iteration = self.last_batch_iteration
        self.last_batch_iteration = 0

        lr_schedule = []
        for i in range(number_of_cycles * self.cycle_size):
            lr_schedule.append(self.get_lr())
            self.last_batch_iteration += 1

        self.last_batch_iteration = store_last_iteration
        lr_schedule = np.array(lr_schedule)
        assert lr_schedule.ndim == 1
        return lr_schedule

    def get_loss_lr_history(self) -> np.ndarray:
        assert self.record_history, "Cannot be used outside of pre-training as loss is not captured"
        return np.array([(loss, *lrs) for loss, lrs in zip(self.loss, self.lrs)])

    def get_averaged_loss_lr_history(self, window=100) -> np.ndarray:
        """ Average loss and learn rate value within window size

        Any remainder outside of window size will not be included in the average returned.

        Mean values returned can be visualised in the follow:

            import matplotlib.pyplot as plt
            plt.scatter(mean_loss_lr[1], mean_loss_lr[0], s=1)
            plt.show()

        Using the graph, determine the base_lr and max_lr.

        Base_lr is the smallest lr value that results in loss decreasing.
        Max_lr is the largest lr before loss becomes unstable.

        """
        assert self.record_history, "Cannot be used outside of pre-training as loss is not captured"
        assert window > 0, "window size cannot be zero or smaller"

        history = self.get_loss_lr_history()

        if window > 1:
            remainder = history.shape[0] % window
            if remainder:
                print(f"last {remainder} rows are exlcuded from average calculation")
                history = history[:-remainder, ...]

            history = history.reshape((-1, window, history.shape[1]))
            history = np.mean(history, axis=1)

        return history
