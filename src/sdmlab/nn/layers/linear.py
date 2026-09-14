# nn/layers/linear.py
from .base import Layer
from ...params.parameter import Parameter
from ...backend import get_backend

class Dense(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        backend = get_backend()
        self.weight = Parameter(backend.random_normal((in_features, out_features), scale=0.01))
        self.bias = Parameter(backend.zeros((out_features,)))

    def forward(self, x):
        return x @ self.weight + self.bias
