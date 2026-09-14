# params/module.py
from .parameter import Parameter

class Module:
    def parameters(self):
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())   # recurse into nested layers
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)        # this is what makes `layer(x)` work instead of `layer.forward(x)`
