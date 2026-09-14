from .function import Function
from ..backend import get_backend

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return get_backend().matmul(a.data, b.data)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = get_backend().matmul(grad_output, b.data.T)
        grad_b = get_backend().matmul(a.data.T, grad_output)
        return grad_a, grad_b

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return get_backend().add(a.data, b.data)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output   # d(a+b)/da = 1, d(a+b)/db = 1 — gradient just passes through


class ReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return get_backend().maximum(x.data, 0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        mask = x.data > 0          # bool array — native numpy/cupy comparison, no backend method needed
        return (grad_output * mask,)   # elementwise multiply — also native, works identically on both backends

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        result = 1 \
            / (1 + get_backend().exp(-x.data))
        ctx.result = result

        return result

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_x = ctx.result
        return (grad_output * sigmoid_x * (1 - sigmoid_x), )

class TanH(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        exp_pos = get_backend().exp(x.data)
        exp_neg = get_backend().exp(-x.data)

        result =  (exp_pos - exp_neg) \
            / (exp_pos + exp_neg) 
        
        ctx.result = result 

        return result

    @staticmethod
    def backward(ctx, grad_output):
        tanh_x = ctx.result

        return (grad_output * (1 - tanh_x ** 2))


class Softmax(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        exp_values = get_backend().exp(x.data - get_backend().max(x.data, axis = -1, keepdims = True))
        result = exp_values \
            / get_backend().sum(exp_values, axis = -1, keepdims = True)

        ctx.result = result

        return result

    @staticmethod
    def backward(ctx, grad_output):
        softmax_x = ctx.result
        return (softmax_x * (grad_output - get_backend().sum(grad_output * softmax_x, axis = -1, keepdims = True)))
