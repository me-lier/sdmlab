import numpy as np

class NumpyBackend:
    def array(self, data):
        return np.asarray(data, dtype=np.float32)

    def matmul(self, a, b):
        return np.matmul(a, b)

    def add(self, a, b):
        return np.add(a, b)

    def zeros(self, shape):
        return np.zeros(shape)

    def random_normal(self, shape, scale=1.0):
        return np.random.normal(loc=0.0, scale=scale, size = shape)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def exp(self, data):
        return np.exp(data)

    def max(self, data, axis = None, keepdims = False):
        return np.max(data, axis = axis, keepdims = keepdims)

    def sum(self, data, axis = None, keepdims = False):
        return np.sum(data, axis = axis, keepdims = keepdims)

    def clip(self, data, min_value, max_value):
        return np.clip(data, min_value, max_value)

    def arange(self, start, stop=None, step=1):
        return np.arange(start, stop, step)

    def mean(self, data, axis=None):
        return np.mean(data, axis=axis)

    def log(self, data):
        return np.log(data)