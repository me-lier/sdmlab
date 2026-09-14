
from .base import Activation      # or just Layer directly — your call whether a separate base is worth it
from ...autograd.ops import TanH as TanHFunction

class TanH(Activation):
    def forward(self, x):
        return TanHFunction.apply(x)
