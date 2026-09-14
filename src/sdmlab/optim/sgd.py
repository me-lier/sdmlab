from .base import Optimizer
from ..backend import get_backend


class VanillaSGD(Optimizer):
    def __init__(self, parameters, lr):
        super().__init__(parameters, lr)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = p.data - lr * p.grad

class SGDWithDecay(Optimizer):
    def __init__(self, parameters, lr, decay):
        super().__init__(parameters, lr)
        self.decay = decay
        self.iteration = 0

    def step(self):
        self.iteration += 1
        for group in self.param_groups:
            lr = group["lr"] / (1 + self.decay * self.iteration)
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = p.data - lr * p.grad