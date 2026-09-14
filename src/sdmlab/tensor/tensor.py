

class Tensor:
    def __init__(self, data, requires_grad=False, _ctx=None):
        from ..backend import get_backend
        backend = get_backend()
        array_type = type(backend.array([]))          # backend-agnostic way to detect "already an array"
        self.data = data if isinstance(data, array_type) else backend.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = _ctx        # None for a leaf; otherwise the Function context that produced this tensor

    @property
    def shape(self):
        return self.data.shape

    def __matmul__(self, other):
        from ..autograd.ops import MatMul     # local import breaks the tensor<->autograd cycle
        return MatMul.apply(self, other)

    def __add__(self, other):
        from ..autograd.ops import Add
        return Add.apply(self, other)

    def backward(self, grad=None):
        from ..backend import get_backend
        if grad is None:
            grad = get_backend().array(1.0)            # seed: d(self)/d(self) = 1

        self.grad = grad if self.grad is None else get_backend().add(self.grad, grad)  # accumulate

        if self._ctx is not None:
            input_grads = type(self._ctx).backward(self._ctx, grad)
            for input_tensor, input_grad in zip(self._ctx.saved_tensors, input_grads):
                if input_tensor.requires_grad:
                    input_tensor.backward(input_grad)
