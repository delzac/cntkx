import cntk as C
import math
import numpy as np


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
        parameter_learners (list): list of cntk learner
        base_lrs (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lrs (float or list): Upper boundaries in the cycle for
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
        pretraining (bool): If True, loss & learn rate will be recorded. get loss lr history functions can then be used.
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

    def __init__(self, parameter_learners, base_lrs=1e-3, max_lrs=6e-3, minibatch_size=None,
                 step_size=2000, lr_policy='triangular2', gamma=0.99994,
                 scale_fn=None, scale_by="cycle", pretraining=False, last_batch_iteration=-1):

        if lr_policy not in ['triangular', 'triangular2', 'exp_range'] and scale_fn is None:
            raise ValueError('lr_policy is invalid and scale_fn is None')

        if scale_by not in ['iteration', 'cycle']:
            raise ValueError("Can only scale by iteration or cycle")

        self.parameter_learners = parameter_learners if isinstance(parameter_learners, list) else [parameter_learners]
        self.base_lrs = base_lrs if isinstance(base_lrs, list) else [base_lrs] * len(self.parameter_learners)
        self.max_lrs = max_lrs if isinstance(max_lrs, list) else [max_lrs] * len(self.parameter_learners)

        assert len(self.base_lrs) == len(self.max_lrs) == len(self.parameter_learners), "number of learners/base_lrs/max_lrs must be equal"

        self.step_size = step_size
        self.minibatch_size = minibatch_size

        self.lr_policy = lr_policy
        self.gamma = gamma
        self.pretraining = pretraining

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
        self.lrs = []
        self.last_batch_iteration = last_batch_iteration
        self.batch_step()
        self.lrs = []

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** x

    def batch_step(self, previous_minibatch_loss=None):
        """
        Updates learners with new learning rate after one training iteration is complete.
        Must be called once for every training iteration/update.
        """

        self.last_batch_iteration += 1
        lrs = self.get_lr()

        # loss and learn rate gets recorded in pre-training mode
        if self.pretraining and previous_minibatch_loss:
            self.loss.append(previous_minibatch_loss)
            self.lrs.append(lrs)

        for learner, lr in zip(self.parameter_learners, lrs):
            learner.reset_learning_rate(C.learning_parameter_schedule(lr, minibatch_size=self.minibatch_size))

        return None

    def get_lr(self):
        """ Get learning rate based on last_batch_iteration count """
        step_size = float(self.step_size)
        cycle = math.floor(1 + self.last_batch_iteration / (2 * step_size))  # Cycle number (int)
        x = abs(self.last_batch_iteration / step_size - 2 * cycle + 1)  # vary between the range of 0 and 1

        lrs = []
        param_lrs = zip(self.base_lrs, self.max_lrs)
        for base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * max(0, (1 - x))
            xx = cycle if self.scale_by == "cycle" else self.last_batch_iteration
            lr = base_lr + base_height * self.scale_fn(xx)
            lrs.append(lr)

        return tuple(lrs)

    def get_lr_schedule(self, number_of_cycles=4):
        """ returns how the learn rate schedule will be like. Useful to check if your
        custom learning policy is working as intended.

        The returned lr_schedule can be visualised in the following:
            import matplotlib.pyplot as plt

            plt.scatter(range(lr_schedule.shape[0]), lr_schedule[:, 0], s=1)  # for first parameter_learner
            plt.show()

            plt.scatter(range(lr_schedule.shape[0]), lr_schedule[:, 1], s=1)  # for second parameter_learner
            plt.show()

        """
        store_last_iteration = self.last_batch_iteration
        self.last_batch_iteration = 0

        lr_schedule = []
        for i in range(number_of_cycles * self.step_size * 2):
            lr_schedule.append(self.get_lr())
            self.last_batch_iteration += 1

        self.last_batch_iteration = store_last_iteration
        lr_schedule = np.array(lr_schedule)
        assert lr_schedule.ndim == 2

        return lr_schedule

    def get_loss_lr_history(self):
        assert self.pretraining, "Cannot be used outside of pre-training as loss is not captured"
        return np.array([(loss, *lrs) for loss, lrs in zip(self.loss, self.lrs)])

    def get_averaged_loss_lr_history(self, window=100):
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
        assert self.pretraining, "Cannot be used outside of pre-training as loss is not captured"
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
