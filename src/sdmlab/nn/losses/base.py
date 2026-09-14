from ...params.module import Module


class Loss(Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError
