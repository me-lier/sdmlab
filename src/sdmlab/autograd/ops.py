from .function import Function
from ..backend import get_backend

def _unbroadcast(grad, target_shape):
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad

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
        a, b = ctx.saved_tensors
        grad_a = _unbroadcast(grad_output, a.data.shape)
        grad_b = _unbroadcast(grad_output, b.data.shape)
        return grad_a, grad_b


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

        return (grad_output * (1 - tanh_x ** 2), )


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
        return (softmax_x * (grad_output - get_backend().sum(grad_output * softmax_x, axis = -1, keepdims = True)), )


class MSE(Function):
    @staticmethod
    def forward(ctx, pred, target):
        ctx.save_for_backward(pred, target)
        diff = pred.data - target.data
        return get_backend().sum(diff ** 2) / diff.size

    @staticmethod
    def backward(ctx, grad_output):
        pred, target = ctx.saved_tensors
        grad_pred = 2 * (pred.data - target.data) / pred.data.size
        return grad_pred * grad_output, None  # No gradient for the target

class CrossEntropy_without_logits(Function):
    @staticmethod
    def forward(ctx, pred, target):
        ctx.save_for_backward(pred, target)
        samples = pred.data.shape[0]
        pred_clipped = get_backend().clip(
            pred.data,
            1e-12,
            1.0 - 1e-12
        )
        if len(target.data.shape) == 1:
            # Class-index encoding
            correct_confidence = pred_clipped[
                get_backend().arange(samples),
                target.data.astype(int)
            ]
        elif len(target.data.shape) == 2:
            # One-hot encoding
            correct_confidence = get_backend().sum(
                pred_clipped * target.data,
                axis=-1
            )
        else:
            raise ValueError("target must be 1D or 2D")
        negative_log_likelihoods = -get_backend().log(correct_confidence)
        return get_backend().mean(negative_log_likelihoods)

    @staticmethod
    def backward(ctx, grad_output):
        pred, target = ctx.saved_tensors
        samples = pred.data.shape[0]
        pred_clipped = get_backend().clip(
            pred.data,
            1e-12,
            1.0 - 1e-12
        )
        if len(target.data.shape) == 1:
            # Class-index encoding
            grad_pred = pred_clipped.copy()
            grad_pred[
                get_backend().arange(samples),
                target.data.astype(int)
            ] -= 1
        elif len(target.data.shape) == 2:
            # One-hot encoding
            grad_pred = pred_clipped - target.data
        else:
            raise ValueError("target must be 1D or 2D")
        return (grad_pred / samples) * grad_output, None  # No gradient for the target


class CrossEntropy(Function):
    """Fused softmax + cross-entropy: takes raw logits, not probabilities.

    Fusing avoids ever building softmax's full Jacobian in backward() — the combined
    gradient collapses to `(probs - target) / samples`, which only holds with respect
    to the pre-softmax logits. A standalone `Softmax` layer feeding a plain log-loss
    would need the real per-element gradient instead; this class assumes logits in.
    """

    @staticmethod
    def forward(ctx, logits, target):
        ctx.save_for_backward(logits, target)
        exp_values = get_backend().exp(
            logits.data - get_backend().max(logits.data, axis=-1, keepdims=True)
        )
        probs = exp_values / get_backend().sum(exp_values, axis=-1, keepdims=True)
        ctx.probs = probs

        samples = logits.data.shape[0]
        probs_clipped = get_backend().clip(probs, 1e-12, 1.0 - 1e-12)
        if len(target.data.shape) == 1:
            # Class-index encoding
            correct_confidence = probs_clipped[
                get_backend().arange(samples),
                target.data.astype(int)
            ]
        elif len(target.data.shape) == 2:
            # One-hot encoding
            correct_confidence = get_backend().sum(
                probs_clipped * target.data,
                axis=-1
            )
        else:
            raise ValueError("target must be 1D or 2D")
        negative_log_likelihoods = -get_backend().log(correct_confidence)
        return get_backend().mean(negative_log_likelihoods, axis=None)

    @staticmethod
    def backward(ctx, grad_output):
        logits, target = ctx.saved_tensors
        probs = ctx.probs
        samples = logits.data.shape[0]
        if len(target.data.shape) == 1:
            # Class-index encoding
            grad_logits = probs.copy()
            grad_logits[
                get_backend().arange(samples),
                target.data.astype(int)
            ] -= 1
        elif len(target.data.shape) == 2:
            # One-hot encoding
            grad_logits = probs - target.data
        else:
            raise ValueError("target must be 1D or 2D")
        return (grad_logits / samples) * grad_output, None  # No gradient for the target