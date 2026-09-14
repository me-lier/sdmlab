
from .base import Activation      # or just Layer directly — your call whether a separate base is worth it
from ...autograd.ops import Sigmoid as SigmoidFunction

class Sigmoid(Activation):
    def forward(self, x):
        return SigmoidFunction.apply(x)
