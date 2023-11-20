import math

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class LRScheduler(_LRScheduler):
    """Code from `https://github.com/NVIDIA/DeepLearningExamples/tree/master/Py
    Torch/LanguageModeling/BERT`"""

    def __init__(self, optimizer, last_epoch=-1):
        # Check if using mixed precision training
        self.mixed_training = False
        base_optimizer = optimizer

        # Check that optimizer param is valid
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        super(LRScheduler, self).__init__(base_optimizer, last_epoch)

    def step(self, epoch=None):
        # Set the current training step
        # ('epoch' is used to be consistent with _LRScheduler)
        if self.mixed_training:
            # The assumption is that the step will be constant
            state_dict = self.optimizer.state[self.optimizer.param_groups[0]
                                              ['params'][0]]
            if 'step' in state_dict:
                self.last_epoch = state_dict['step'] + 1
            else:
                self.last_epoch = 1
        else:
            self.last_epoch = epoch if epoch is not None else self.last_epoch + 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class ConstantLR(LRScheduler):

    def get_lr(self):
        return self.base_lrs


class CosineWarmUpScheduler(LRScheduler):
    """Applies a warm up period to the learning rate."""

    def __init__(self, optimizer, warmup, total_steps, last_epoch=-1):
        self.warmup = warmup
        self.total_steps = total_steps
        super(CosineWarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        if progress < self.warmup:
            return [
                base_lr * progress / self.warmup for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr * (0.5 * (1.0 + torch.cos(math.pi + progress)))
                for base_lr in self.base_lrs
            ]


class ConstantWarmUpScheduler(LRScheduler):
    """Applies a warm up period to the learning rate."""

    def __init__(self, optimizer, warmup, total_steps, last_epoch=-1):
        self.warmup = warmup
        self.total_steps = total_steps
        super(ConstantWarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        if progress < self.warmup:
            return [
                base_lr * progress / self.warmup for base_lr in self.base_lrs
            ]
        else:
            return self.base_lrs


class LinearWarmUpScheduler(LRScheduler):
    """Applies a warm up period to the learning rate."""

    def __init__(self, optimizer, warmup, total_steps, last_epoch=-1):
        self.warmup = warmup
        self.total_steps = total_steps
        super(LinearWarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        if progress < self.warmup:
            return [
                base_lr * progress / self.warmup for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr * max((progress - 1.0) / (self.warmup - 1.0), 0.)
                for base_lr in self.base_lrs
            ]


class PolyWarmUpScheduler(LRScheduler):
    """Applies a warm up period to the learning rate."""

    def __init__(self,
                 optimizer,
                 warmup,
                 total_steps,
                 degree=0.5,
                 last_epoch=-1,
                 base_lr=1.,
                 device='cpu'):
        self.warmup = torch.tensor(warmup, device=device)
        self.total_steps = torch.tensor(total_steps, device=device)
        self.degree = torch.tensor(degree, device=device)
        device_last_epoch = torch.tensor(last_epoch, device=device)
        self.base_lr = torch.tensor(base_lr, device=device)
        self.device = device
        super(PolyWarmUpScheduler, self).__init__(optimizer, device_last_epoch)

    def step(self, epoch=None):
        param_group = self.optimizer.param_groups[0]
        if 'step' in param_group:
            self.last_epoch = param_group['step'] + 1
        else:
            self.last_epoch = torch.tensor(1., device=self.device)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        lr_tensor = torch.where(progress < self.warmup,
                                self.base_lr * progress / self.warmup,
                                self.base_lr * ((1.0 - progress)**self.degree))
        return [lr_tensor for _ in range(len(self.optimizer.param_groups))]
