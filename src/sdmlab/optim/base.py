
class Optimizer:
    def __init__(self, parameters, lr):
        self.param_groups = [{"params": list(parameters), "lr": lr}]
        self.state = {}   # per-parameter buffers (momentum velocity, Adam moments, ...) — keyed by id(param)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None

    def step(self):
        raise NotImplementedError
