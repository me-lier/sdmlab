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