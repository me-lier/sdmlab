
from .base import Activation      # or just Layer directly — your call whether a separate base is worth it
from ...autograd.ops import ReLU as ReLUFunction

class ReLU(Activation):
    def forward(self, x):
        return ReLUFunction.apply(x)
