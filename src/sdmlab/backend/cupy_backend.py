import cupy as np

class CupyBackend:
    def array(self, data):
        return np.asarray(data, dtype=np.float32)

    def matmul(self, a, b):
        return np.matmul(a, b)

    def add(self, a, b):
        return np.add(a, b)

    def zeros(self, shape):
        return np.zeros(shape)

    def random_normal(self, shape, scale=1.0):
        return np.random.normal(loc = 0.0, scale = scale, size = shape)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def exp(self, data):
        return np.exp(data)

    def max(self, data, axis = -1, keepdims = False):
            return np.max(data, axis = axis, keepdims = keepdims)

    def sum(self, data, axis = -1, keepdims = False):
            return np.sum(data, axis = axis, keepdims = keepdims)