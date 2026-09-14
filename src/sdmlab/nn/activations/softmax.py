
from .base import Activation      # or just Layer directly — your call whether a separate base is worth it
from ...autograd.ops import Softmax as SoftmaxFunction

class Softmax(Activation):
    def forward(self, x):
        return SoftmaxFunction.apply(x)
