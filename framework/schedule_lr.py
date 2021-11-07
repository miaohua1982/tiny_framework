from collections import Counter

class LRScheduler(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.last_epoch = 0

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma):
        assert step_size>0, 'In StepLR, the value of step size must greater than zero'
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > 0 and self.last_epoch % self.step_size == 0:
            return self.optimizer.get_lr()*self.gamma
        else:
            return self.optimizer.get_lr()

    def step(self):
        self.last_epoch += 1
        lr = self.get_lr()
        self.optimizer.set_lr(lr)

class MultiStepLR(LRScheduler):
    def __init__(self, optimizer, milestones, gamma):
        super(LRScheduler, self).__init__(optimizer)
        self.milestones = milestones
        self.gamma = gamma

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return self.optimizer.get_lr()
        else:
            return self.optimizer.get_lr()*self.gamma

    def step(self):
        self.last_epoch += 1
        lr = self.get_lr()
        self.optimizer.set_lr(lr)
