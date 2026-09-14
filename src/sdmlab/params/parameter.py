

from ..tensor.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)   # a Parameter is always a leaf, always trainable
