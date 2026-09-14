from ...params.module import Module


class Activation(Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError
