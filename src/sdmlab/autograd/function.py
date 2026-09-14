

class Function:
    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    @classmethod
    def apply(cls, *inputs):
        ctx = cls()
        raw_output = cls.forward(ctx, *inputs)
        requires_grad = any(t.requires_grad for t in inputs)
        from ..tensor.tensor import Tensor
        return Tensor(raw_output, requires_grad=requires_grad, _ctx=ctx if requires_grad else None)
